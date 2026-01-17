
// --- Constants & Structs ---
const PI = 3.14159265359;
const MAX_RESERVOIR_M_CANDIDATES = 32u;
const MAX_RESERVOIR_M_TEMPORAL = 20u;

struct Reservoir {
    y: u32,       // Light Index
    w_sum: f32,   // Sum of weights
    M: u32,       // Number of samples seen
    W: f32,       // Generalized weight
}
// Struct layout should match Rust. standard layout 16 bytes alignment.
// y: vec4f (16)
// w_sum: f32 (4)
// M: u32 (4)
// W: f32 (4)
// light_index: u32 (4) -> Offset 28 -> Pad to 32?
// Total 32 bytes fits nicely.

struct Light {
    position: vec3f,
    type_: u32,
    u: vec3f,
    area: f32,
    v: vec3f,
    pad: u32,
    emission: vec4f,
}

struct Camera {
    view_proj: array<vec4f, 4>,
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    view_pos: vec4f,
    prev_view_proj: array<vec4f, 4>,
    frame_count: u32,
    num_lights: u32,
}

// --- Bindings ---

// Group 0: G-Buffer & Scene context (Lights, Camera)
@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_motion: texture_2d<f32>; // For temporal reuse

@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<uniform> scene_info: vec4u; // x=light_count, y=frame_count

// Prev G-Buffer for validation
@group(0) @binding(7) var prev_gbuffer_pos: texture_2d<f32>;
@group(0) @binding(8) var prev_gbuffer_normal: texture_2d<f32>;

// Group 1: Reservoirs
@group(1) @binding(0) var<storage, read_write> prev_reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read_write> curr_reservoirs: array<Reservoir>;

// --- Utilities ---
// PCG Hash for better quality random numbers
fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

fn is_valid_neighbor(
    curr_pos: vec3f, curr_normal: vec3f, curr_mat: u32,
    prev_pos: vec3f, prev_normal: vec3f, prev_mat: u32,
    camera_pos: vec3f,
) -> bool {
    // 1. マテリアルIDが違うなら別人
    if curr_mat != prev_mat { return false; }

    // 2. 法線の向きが違いすぎたらNG
    if dot(curr_normal, prev_normal) < 0.9 { return false; }

    // 3. 位置が離れすぎていたらNG
    let dist_diff_sq = dot(curr_pos - prev_pos, curr_pos - prev_pos);

    // カメラからその点までの距離
    let dist_to_camera_sq = dot(curr_pos - camera_pos, curr_pos - camera_pos);

    // 許容誤差を「カメラ距離の 2%」などに設定
    // シーンのスケールに合わせて 0.01 ~ 0.05 くらいで調整してください
    let threshold_ratio = 0.03; 
    
    // 最低保証値 (0.01) を入れておくと、至近距離で厳しすぎるのを防げます
    let threshold = max(0.01, dist_to_camera_sq * threshold_ratio);

    if dist_diff_sq > threshold { return false; }
    return true;
}

// Reuse update_reservoir from raytrace.wgsl idea
fn update_reservoir(r: ptr<function, Reservoir>, light_idx: u32, w: f32, rnd: f32) -> bool {
    (*r).w_sum += w;
    (*r).M += 1u;
    if rnd * (*r).w_sum < w {
        (*r).y = light_idx;
        return true;
    }
    return false;
}

// Calculate target PDF (p_hat) - Unshadowed Luminance
fn target_pdf(light_idx: u32, pos: vec3f, normal: vec3f) -> f32 {
    let light = lights[light_idx];
    let L_vec = light.position - pos;
    let dist_sq = dot(L_vec, L_vec);
    let dist = sqrt(dist_sq);
    let L = L_vec / dist;

    let NdotL = max(dot(normal, L), 0.0);
    // Simple point light attenuation
    let attenuation = 1.0 / max(dist_sq, 0.01);
    let luminance = dot(light.emission.rgb, vec3f(0.2126, 0.7152, 0.0722)) * light.emission.w;
    return luminance * attenuation * NdotL * light.area;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(gbuffer_pos);
    if id.x >= size.x || id.y >= size.y { return; }

    let pixel_idx = id.y * size.x + id.x;
    // Initialize seed with PCG hash of coordinates and frame
    let seed_init = id.y * size.x + id.x + scene_info.y * 0x9e3779b9u;
    let seed = pcg_hash(seed_init);

    // Read G-Buffer
    let coord = vec2<i32>(id.xy);
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    let normal_w = textureLoad(gbuffer_normal, coord, 0);

    if pos_w.w < 0.0 {
        // Background - clear reservoir
        curr_reservoirs[pixel_idx].w_sum = 0.0;
        curr_reservoirs[pixel_idx].W = 0.0;
        return;
    }

    let pos = pos_w.xyz;
    let normal = normal_w.xyz;
    let mat_id = i32(pos_w.w + 0.5);
    let num_lights = scene_info.x;

    var r: Reservoir;
    r.w_sum = 0.0;
    r.M = 0u;
    r.W = 0.0;
    r.y = 0u;

    // --- Phase 1: Initial Candidate Search (RIS) ---
    // Generate M candidates
    let M_candidates = 8u; // Number of candidates per pixel
    for (var i = 0u; i < M_candidates; i++) {
        let rnd_light = rand_float(seed + i * 1143u);
        let light_idx = min(u32(rnd_light * f32(num_lights)), num_lights - 1u);

        let p_hat = target_pdf(light_idx, pos, normal);
        let source_pdf = 1.0 / f32(num_lights); // Uniform sampling
        let w = p_hat / source_pdf;

        update_reservoir(&r, light_idx, w, rand_float(seed + i * 7919u));
    }
    
    // Compute W for the initial reservoir
    let p_hat_final = target_pdf(r.y, pos, normal);
    if p_hat_final > 0.0 {
        r.W = (1.0 / p_hat_final) * (r.w_sum / f32(r.M));
    } else {
        r.W = 0.0;
    }
    
    // Clamp history
    if r.M > MAX_RESERVOIR_M_TEMPORAL {
        r.M = MAX_RESERVOIR_M_TEMPORAL;
    }
    // --- Phase 2: Temporal Reuse ---
    let motion = textureLoad(gbuffer_motion, coord, 0);
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
    let prev_uv = uv + motion.xy;
    let prev_id_xy = vec2u(prev_uv * vec2f(size));
    let prev_pixel_idx = prev_id_xy.y * size.x + prev_id_xy.x;

    // 画面外チェック
    if prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0 {

        // 過去のG-Bufferを読んでチェック
        let prev_pos_data = textureLoad(prev_gbuffer_pos, prev_id_xy, 0);
        let prev_normal_data = textureLoad(prev_gbuffer_normal, prev_id_xy, 0);

        let prev_mat_id = bitcast<u32>(prev_pos_data.w);

        if is_valid_neighbor(pos, normal, u32(pos_w.w + 0.5), prev_pos_data.xyz, prev_normal_data.xyz, prev_mat_id, camera.view_pos.xyz) {
            // ★合格！過去のReservoirをマージ
            var prev_r = prev_reservoirs[prev_pixel_idx];
            
            // 過去のReservoirが持つライトが、今の場所(pos)でどれくらい明るいか再評価
            // (これをしないと、影に入ったのに明るいままになる)
            let p_hat_prev = target_pdf(prev_r.y, pos, normal);
            
            // マージ処理 (Algorithm 3 in ReSTIR paper)
            // 過去の重み補正: limit M to avoid history explosion (max 20)
            prev_r.M = min(prev_r.M, 20u); 
            
            // update_reservoirに渡すweightは `p_hat * W * M`
            let w_prev = p_hat_prev * prev_r.W * f32(prev_r.M);

            update_reservoir(&r, prev_r.y, w_prev, rand_float(seed + M_candidates * 7919u + 1u));
            
            // サンプル数(M)を加算
            r.M += prev_r.M;
        }
    }
    
    // Finalize W
    let pf = target_pdf(r.y, pos, normal);
    if pf > 0.0 {
        r.W = (1.0 / pf) * (r.w_sum / f32(r.M));
    } else {
        r.W = 0.0;
    }
    
    // Write new reservoir
    curr_reservoirs[pixel_idx] = r;
}
