@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var pos_tex: texture_2d<f32>;
@group(0) @binding(6) var motion_tex: texture_2d<f32>; // New Binding


struct PostParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
};
@group(0) @binding(5) var<uniform> params: PostParams;

fn gauss(x: f32, sigma: f32) -> f32 {
    if sigma < 0.001 {
        return select(0.0, 1.0, abs(x) < 0.001);
    }
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

fn decode_octahedral_normal(e: vec2f) -> vec3f {
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    let coords = id.xy;
    let idx = coords.y * params.width + coords.x;

    let center_color = textureLoad(raw_tex, coords, 0).rgb;
    let center_normal_encoded = textureLoad(normal_tex, coords, 0).xy;
    let center_normal = decode_octahedral_normal(center_normal_encoded);
    let center_pos = textureLoad(pos_tex, coords, 0).xyz;

    var sum_color = vec3f(0.0);
    var sum_weight = 0.0;

    // Bilateral Filter Parameters
    let sigma_spatial = 1.5;
    let sigma_color = 0.5;
    let sigma_normal = 0.1;
    let sigma_pos = 0.1;
    let kernel_radius = 2;

    for (var dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (var dx = -kernel_radius; dx <= kernel_radius; dx++) {
            let offset = vec2i(dx, dy);
            let sample_coords = vec2i(coords) + offset;

            // Bounds check
            if sample_coords.x < 0 || sample_coords.y < 0 || sample_coords.x >= i32(params.width) || sample_coords.y >= i32(params.height) {
                continue;
            }

            let sample_color = textureLoad(raw_tex, sample_coords, 0).rgb;
            let sample_normal_encoded = textureLoad(normal_tex, sample_coords, 0).xy;
            let sample_normal = decode_octahedral_normal(sample_normal_encoded);
            let sample_pos = textureLoad(pos_tex, sample_coords, 0).xyz;

            // Spatial weight
            let dist_spatial = length(vec2f(offset));
            let w_spatial = gauss(dist_spatial, sigma_spatial);

            // Color weight (similarity)
            let dist_color = length(sample_color - center_color);
            let w_color = gauss(dist_color, sigma_color);

            // Normal weight
            let dot_normal = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
            let w_normal = pow(dot_normal, 20.0);

            // Position weight (depth/plane)
            let dist_pos = length(sample_pos - center_pos);
            let w_pos = gauss(dist_pos, sigma_pos);

            let weight = w_spatial * w_color * w_normal * w_pos;

            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    var filtered_color = center_color;
    if sum_weight > 0.001 {
        filtered_color = sum_color / sum_weight;
    }

    // --- TAA (Reprojection) ---
    // Calculate Mean/Min/Max of 3x3 neighborhood for clamping
    var c_min = vec3f(1.0e20);
    var c_max = vec3f(-1.0e20);
    var c_avg = vec3f(0.0);
    var count = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let offset = vec2i(dx, dy);
            let s_coords = vec2i(coords) + offset;
            if s_coords.x >= 0 && s_coords.y >= 0 && s_coords.x < i32(params.width) && s_coords.y < i32(params.height) {
                let s_col = textureLoad(raw_tex, s_coords, 0).rgb;
                c_min = min(c_min, s_col);
                c_max = max(c_max, s_col);
                c_avg += s_col;
                count += 1.0;
            }
        }
    }
    c_avg /= count;

    // Temporal Reuse
    var history_color = filtered_color;
    var valid_history = false;

    if params.frame_count > 0u {
        let motion = textureLoad(motion_tex, coords, 0).xy;
        let uv = (vec2f(coords) + 0.5) / vec2f(f32(params.width), f32(params.height));
        let prev_uv = uv + motion;
        let prev_coords = vec2i(prev_uv * vec2f(f32(params.width), f32(params.height)) - 0.5);

        if prev_coords.x >= 0 && prev_coords.y >= 0 && prev_coords.x < i32(params.width) && prev_coords.y < i32(params.height) {
            let prev_idx = u32(prev_coords.y) * params.width + u32(prev_coords.x);
            let hist_val = accumulation[prev_idx];
            // history stores (accumulated_color, sample_count). 
            // We want the average color for TAA history. 
            // Previous implementation stored SUM. 
            // NOW we store the filtered color directly (last frame's result).

            history_color = hist_val.rgb;
            valid_history = true;
        }
    }

    var final_color = filtered_color;

    if valid_history {
        // Clamp History (Simple AABB)
        let clamped_history = clamp(history_color, c_min, c_max);

        let blend_factor = 0.9; // 90% history, 10% new
        final_color = mix(final_color, clamped_history, blend_factor);
    }
    
    // Store result in accumulation buffer for next frame
    accumulation[idx] = vec4f(final_color, 1.0);

    // Gamma correction (Manual)
    let display_color = pow(final_color, vec3f(1.0 / 2.2));

    textureStore(out_tex, vec2i(coords), vec4f(display_color, 1.0));
}
