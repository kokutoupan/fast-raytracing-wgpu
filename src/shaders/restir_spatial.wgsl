// --- Constants & Structs ---
const PI = 3.14159265359;

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
@group(1) @binding(0) var<storage, read> in_reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read_write> out_reservoirs: array<Reservoir>;

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

fn decode_octahedral_normal(e: vec2f) -> vec3f {
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
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
    // Mixing x, y, and frame to avoid correlations
    let seed_init = id.y * size.x + id.x + scene_info.y * 0x9e3779b9u;
    let seed = pcg_hash(seed_init);

    // Read G-Buffer
    let coord = vec2<i32>(id.xy);
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    let normal_w = textureLoad(gbuffer_normal, coord, 0);

    if pos_w.w < 0.0 {
        // Background - clear reservoir
        out_reservoirs[pixel_idx].w_sum = 0.0;
        out_reservoirs[pixel_idx].W = 0.0;
        return;
    }

    let pos = pos_w.xyz;
    let normal = decode_octahedral_normal(normal_w.xy);
    let mat_id = i32(pos_w.w + 0.5);
    let num_lights = scene_info.x;

    // --- Phase 3: Spatial Reuse ---
    let MAX_SPATIAL_SAMPLES = 4u;
    let spatial_radius = 30.0; // Screen space radius

    // Initialize with temporal result (binding 0 is Input, binding 1 is Output)
    var current_r = in_reservoirs[pixel_idx];
    
    // Create a new combining reservoir
    var spatial_r: Reservoir;
    spatial_r.w_sum = 0.0;
    spatial_r.M = 0u;
    spatial_r.W = 0.0;
    spatial_r.y = 0u;

    // 1. Merge current (Temporal) reservoir
    let p_hat_curr = target_pdf(current_r.y, pos, normal);
    let w_curr = p_hat_curr * current_r.W * f32(current_r.M);
    update_reservoir(&spatial_r, current_r.y, w_curr, rand_float(seed + 1234u));
    spatial_r.M += current_r.M;

    // 2. Merge neighbors
    for (var i = 0u; i < MAX_SPATIAL_SAMPLES; i++) {
        let rnd_offset = vec2f(rand_float(seed + i * 13u), rand_float(seed + i * 17u));
        let offset = (rnd_offset * 2.0 - 1.0) * spatial_radius;
        let neighbor_coord = vec2<i32>(vec2f(coord) + offset);

        // Bounds check
        if neighbor_coord.x < 0 || neighbor_coord.x >= i32(size.x) || neighbor_coord.y < 0 || neighbor_coord.y >= i32(size.y) {
            continue;
        }
        
        // Neighbor geometric validation (using current G-Buffer, since spatial reuse is same frame)
        let neighbor_pos_w = textureLoad(gbuffer_pos, neighbor_coord, 0);
        let neighbor_normal_w = textureLoad(gbuffer_normal, neighbor_coord, 0);

        if neighbor_pos_w.w < 0.0 { continue; }

        let neighbor_pos = neighbor_pos_w.xyz;
        let neighbor_normal = decode_octahedral_normal(neighbor_normal_w.xy);
        let neighbor_mat_id = u32(neighbor_pos_w.w + 0.5);

        if !is_valid_neighbor(pos, normal, u32(pos_w.w + 0.5), neighbor_pos, neighbor_normal, neighbor_mat_id, camera.view_pos.xyz) {
           continue;
        }

        let neighbor_idx = u32(neighbor_coord.y) * size.x + u32(neighbor_coord.x);
        var neighbor_r = in_reservoirs[neighbor_idx]; // Read validation from same input buffer
        
        // Re-evaluate neighbor light at current surface
        let p_hat_neighbor = target_pdf(neighbor_r.y, pos, normal);
        
        // Clamp neighbor M
        neighbor_r.M = min(neighbor_r.M, 20u); // Limit spatial influence

        let w_neighbor = p_hat_neighbor * neighbor_r.W * f32(neighbor_r.M);

        update_reservoir(&spatial_r, neighbor_r.y, w_neighbor, rand_float(seed + i * 999u));
        spatial_r.M += neighbor_r.M;
    }

    // Finalize W
    let pf = target_pdf(spatial_r.y, pos, normal);
    if pf > 0.0 {
        spatial_r.W = (1.0 / pf) * (spatial_r.w_sum / f32(spatial_r.M));
    } else {
        spatial_r.W = 0.0;
    }

    out_reservoirs[pixel_idx] = spatial_r;
}
