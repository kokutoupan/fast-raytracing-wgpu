
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
    view_proj: mat4x4f,
    view_inv: mat4x4f,
    proj_inv: mat4x4f,
    view_pos: vec4f,
    prev_view_proj: mat4x4f,
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

// Group 1: Reservoirs
@group(1) @binding(0) var<storage, read_write> prev_reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read_write> curr_reservoirs: array<Reservoir>;

// --- Utilities ---
fn simple_rand(seed: u32) -> f32 {
    var x = seed;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return f32(x) / 4294967296.0;
}

fn tea(v0: u32, v1: u32) -> u32 {
    var s0 = 0u;
    var v0_ = v0;
    var v1_ = v1;
    for (var n = 0; n < 4; n++) {
        s0 += 0x9e3779b9u;
        v0_ += ((v1_ << 4u) + 0xa341316cu) ^ (v1_ + s0) ^ ((v1_ >> 5u) + 0xc8013ea4u);
        v1_ += ((v0_ << 4u) + 0xad90777du) ^ (v0_ + s0) ^ ((v0_ >> 5u) + 0x7e95761eu);
    }
    return v0_;
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
    let seed = tea(id.x + id.y * size.x, scene_info.y);

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
    let num_lights = scene_info.x;

    var r: Reservoir;
    r.w_sum = 0.0;
    r.M = 0u;
    r.W = 0.0;
    r.y = 0u;

    // --- Phase 1: Initial Candidate Search (RIS) ---
    // Generate M candidates
    let M_candidates = 4u; // Number of candidates per pixel
    for (var i = 0u; i < M_candidates; i++) {
        let rnd_light = simple_rand(seed + i * 1143u);
        let light_idx = min(u32(rnd_light * f32(num_lights)), num_lights - 1u);

        let p_hat = target_pdf(light_idx, pos, normal);
        let source_pdf = 1.0 / f32(num_lights); // Uniform sampling
        let w = p_hat / source_pdf;

        update_reservoir(&r, light_idx, w, simple_rand(seed + i * 7919u));
    }
    
    // Compute W for the initial reservoir
    let p_hat_final = target_pdf(r.y, pos, normal);
    if p_hat_final > 0.0 {
        r.W = (1.0 / p_hat_final) * (r.w_sum / f32(r.M));
    } else {
        r.W = 0.0;
    }
    
    // --- Phase 2: Temporal Reuse ---
    // Note: User asked for "without motion" first, but motion vector texture is bound.
    // Let's implement basic logic. If no motion used, we just read same pixel index from prev?
    // No, camera moves, so we MUST reuse based on reprojection or at least same screen coord if static.
    // "temporal reuse without motion" might mean "assume motion is 0" or "don't use motion vectors yet".
    // But since I implemented motion vectors, I should use them if I can.
    // The user specifically said "time reuse without motion" (motionなしで).
    // So I will just use current coord `id.xy` to read from prev_reservoirs for now.
    // This assumes static camera/scene or ghosting will appear.
    
    // Clamp history
    if r.M > MAX_RESERVOIR_M_TEMPORAL {
        r.M = MAX_RESERVOIR_M_TEMPORAL;
    }

    let prev_pixel_idx = pixel_idx; // No reproject
    let prev_r = prev_reservoirs[prev_pixel_idx];
    
    // Combine
    // Update reservoir takes weight = p_hat * W * M
    // But here p_hat is relative to current shading point.
    // We need to re-evaluate p_hat of the previous sample at current position (shift mapping? or just reuse p_hat if ignoring geometric changes).
    // Proper way: p_hat = target_pdf(prev_r.light_index, pos, normal);
    // weight = p_hat * prev_r.W * prev_r.M;

    let prev_p_hat = target_pdf(prev_r.y, pos, normal);
    let prev_weight = prev_p_hat * prev_r.W * f32(prev_r.M);

    if update_reservoir(&r, prev_r.y, prev_weight, simple_rand(seed + 9999u)) {
        r.M += prev_r.M;
    } else {
        r.M += prev_r.M;
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
