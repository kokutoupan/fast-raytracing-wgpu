@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> history: array<vec4f>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var pos_tex: texture_2d<f32>;
@group(0) @binding(6) var motion_tex: texture_2d<f32>; // New Binding
@group(0) @binding(7) var<storage, read_write> accumulation: array<vec4f>; // New Binding


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

fn rgb_to_ycocg(rgb: vec3f) -> vec3f {
    let y = dot(rgb, vec3f(0.25, 0.5, 0.25));
    let co = dot(rgb, vec3f(0.5, 0.0, -0.5));
    let cg = dot(rgb, vec3f(-0.25, 0.5, -0.25));
    return vec3f(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3f) -> vec3f {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    return vec3f(y + co - cg, y + cg, y - co - cg);
}

// --- Helper for Reversible Tonemap ---
fn resolve_tonemap(c: vec3f) -> vec3f {
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}

fn resolve_inverse_tonemap(c: vec3f) -> vec3f {
    return c / (1.0 - max(c.r, max(c.g, c.b)));
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

    // 2. Neighborhood Sampling (3x3) & AABB Calculation in YCoCg
    
    // Sample accumulated color? No, we filter the raw frame first (filtered_color).
    // We want to constrain history to the CURRENT frame's neighborhood.
    // 3. Neighborhood Sampling (Variance Clipping)
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    
    // Tonemap center (filtered_color is HDR)
    let tm_filtered = resolve_tonemap(filtered_color);
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
             let neighbor_coords = vec2i(coords) + vec2i(dx, dy);
             var s_col = vec3f(0.0);
             if neighbor_coords.x >= 0 && neighbor_coords.y >= 0 && neighbor_coords.x < i32(params.width) && neighbor_coords.y < i32(params.height) {
                 s_col = textureLoad(raw_tex, neighbor_coords, 0).rgb;
             } else {
                 s_col = filtered_color;
             }
             
             // Tonemap neighbor
             let s_tm = resolve_tonemap(s_col);
             let s_ycocg = rgb_to_ycocg(s_tm);
             
             m1 += s_ycocg;
             m2 += s_ycocg * s_ycocg;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;

    let sigma = sqrt(max(vec3f(0.0), m2 - m1 * m1));
    let gamma = 1.5; // Higher Gamma for stability
    let c_min = m1 - gamma * sigma;
    let c_max = m1 + gamma * sigma;
    let c_avg = m1; 

    // 3. History Sampling & Reprojection
    var history_color = tm_filtered; // Use Tonemapped filtered as default
    var valid_history = false;
    let blend_factor_base = 0.90; // High history weight to kill micro-shimmer
    var blend_factor = blend_factor_base;
    var structure_motion = vec2f(0.0);

    if params.frame_count > 0u {
        structure_motion = textureLoad(motion_tex, coords, 0).xy;

        let uv = (vec2f(coords) + 0.5) / vec2f(f32(params.width), f32(params.height));
        let prev_uv = uv + structure_motion; 

        let prev_pos = prev_uv * vec2f(f32(params.width), f32(params.height)) - 0.5;
        // ... (Code continues below)

        let p0 = vec2i(floor(prev_pos));
        let p1 = p0 + vec2i(1, 0);
        let p2 = p0 + vec2i(0, 1);
        let p3 = p0 + vec2i(1, 1);

        let f = fract(prev_pos);
        
        let margin = 0.0; // Strict check
        if prev_uv.x >= 0.0 && prev_uv.y >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y <= 1.0 {
             let w_i = i32(params.width);
             let h_i = i32(params.height);
             
             let p0_c = clamp(p0, vec2i(0), vec2i(w_i - 1, h_i - 1));
             let p1_c = clamp(p1, vec2i(0), vec2i(w_i - 1, h_i - 1));
             let p2_c = clamp(p2, vec2i(0), vec2i(w_i - 1, h_i - 1));
             let p3_c = clamp(p3, vec2i(0), vec2i(w_i - 1, h_i - 1));

             let idx0 = u32(p0_c.y) * params.width + u32(p0_c.x);
             let idx1 = u32(p1_c.y) * params.width + u32(p1_c.x);
             let idx2 = u32(p2_c.y) * params.width + u32(p2_c.x);
             let idx3 = u32(p3_c.y) * params.width + u32(p3_c.x);

             // Store directly to temp variables to avoid mutability confusion, or just use c0, c1...
             // Since history is linear/HDR, we need to tonemap it before mixing.
             let c0 = resolve_tonemap(history[idx0].rgb);
             let c1 = resolve_tonemap(history[idx1].rgb);
             let c2 = resolve_tonemap(history[idx2].rgb);
             let c3 = resolve_tonemap(history[idx3].rgb);

              let c01 = mix(c0, c1, f.x);
              let c23 = mix(c2, c3, f.x);
              history_color = mix(c01, c23, f.y);
              valid_history = true;
        }
    }

    // We work in Tonemapped space now
    var final_tm = tm_filtered;

    if valid_history {
        // 4. History Rectification (Variance Clipping) - In Tonemapped Space
        let hist_ycocg = rgb_to_ycocg(history_color);
        
        let clipped_ycocg = clamp(hist_ycocg, c_min, c_max);
        
        let clamped_history = ycocg_to_rgb(clipped_ycocg);

        let motion_px = structure_motion * vec2f(f32(params.width), f32(params.height));
        let speed = length(motion_px);

        let blend_factor = mix(0.97, 0.85, clamp(speed, 0.0, 1.0));
        
        // Feedback
        final_tm = mix(tm_filtered, clamped_history, blend_factor);
    }
    
    // Inverse Tonemap to get back to Linear HDR
    var final_color = resolve_inverse_tonemap(final_tm);
    final_color = max(vec3f(0.0), final_color); // Safety
    


    // Store result
    accumulation[idx] = vec4f(final_color, 1.0);

    // Gamma correction
    let display_color = pow(final_color, vec3f(1.0 / 2.2));

    textureStore(out_tex, vec2i(coords), vec4f(display_color, 1.0));
}
