@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> history: array<vec4f>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var pos_tex: texture_2d<f32>;
@group(0) @binding(6) var motion_tex: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(8) var smp: sampler; // New Binding
@group(0) @binding(9) var albedo_tex: texture_2d<f32>; // Albedo Binding

struct PostParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
    jitter: vec2f,
    padding: vec2f,
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

// Manual Bilinear Sampling helper removed (using Rgba16Float and hardware filtering)

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    let coords = id.xy;
    let idx = coords.y * params.width + coords.x;
    let size = vec2f(f32(params.width), f32(params.height));
    let uv = (vec2f(coords) + 0.5) / size;

    // Unjitter Offset (matches User's logic: X inverted, Y kept, Scale 0.5)
    let unjitter_offset = vec2f(-params.jitter.x, params.jitter.y) * 0.5;
    let sample_uv = uv + unjitter_offset;

    // Use Hardware Bilinear Sampling (now supported with Rgba16Float)
    let center_color = textureSampleLevel(raw_tex, smp, sample_uv, 0.0).rgb;
    let center_albedo = textureSampleLevel(albedo_tex, smp, sample_uv, 0.0).rgb;
    
    // For Normals/Pos, keep using Load (Unjittered sampling for G-Buffer? Maybe later, sticking to center for now)
    let center_normal_encoded = textureLoad(normal_tex, coords, 0).xy;
    let center_normal = decode_octahedral_normal(center_normal_encoded);
    let center_pos = textureLoad(pos_tex, coords, 0).xyz;

    var sum_color = vec3f(0.0);
    var sum_weight = 0.0;

    // Bilateral Filter Parameters
    let sigma_spatial = 1.5;
    let sigma_color = 0.4;
    let sigma_normal = 0.1;
    let sigma_pos = 0.1;
    let kernel_radius = 2;

    for (var dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (var dx = -kernel_radius; dx <= kernel_radius; dx++) {
            let offset = vec2i(dx, dy);
            let neighbor_coords = vec2i(coords) + offset;
            let neighbor_uv = (vec2f(neighbor_coords) + 0.5) / size;
            let neighbor_sample_uv = neighbor_uv + unjitter_offset;

            // Bounds check (pixel coords)
            if neighbor_coords.x < 0 || neighbor_coords.y < 0 || neighbor_coords.x >= i32(params.width) || neighbor_coords.y >= i32(params.height) {
                continue;
            }

            // Unjittered Sample
            let sample_color = textureSampleLevel(raw_tex, smp, neighbor_sample_uv, 0.0).rgb;
            let sample_albedo = textureSampleLevel(albedo_tex, smp, neighbor_sample_uv, 0.0).rgb;
            
            let sample_normal_encoded = textureLoad(normal_tex, neighbor_coords, 0).xy;
            let sample_normal = decode_octahedral_normal(sample_normal_encoded);
            let sample_pos = textureLoad(pos_tex, neighbor_coords, 0).xyz;

            // Spatial weight
            let dist_spatial = length(vec2f(offset));
            let w_spatial = gauss(dist_spatial, sigma_spatial);

            // Color weight (similarity) using Albedo
            let dist_color = length(sample_albedo - center_albedo); 
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

    // --- Variance Clipping ---
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    
    // Tonemap center (filtered_color is HDR)
    let tm_filtered = resolve_tonemap(filtered_color);
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
             let neighbor_coords = vec2i(coords) + vec2i(dx, dy);
             let neighbor_uv = (vec2f(neighbor_coords) + 0.5) / size;
             let neighbor_sample_uv = neighbor_uv + unjitter_offset;

             var s_col = vec3f(0.0);
             if neighbor_coords.x >= 0 && neighbor_coords.y >= 0 && neighbor_coords.x < i32(params.width) && neighbor_coords.y < i32(params.height) {
                 s_col = textureSampleLevel(raw_tex, smp, neighbor_sample_uv, 0.0).rgb;
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
    let gamma = 1.25; // Slightly deeper box
    let c_min = m1 - gamma * sigma;
    let c_max = m1 + gamma * sigma;
    let c_avg = m1; 

    // 3. History Sampling & Reprojection
    var history_color = tm_filtered; // Use Tonemapped filtered as default
    var valid_history = false;
    let blend_factor_base = 0.9;
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

        if prev_uv.x >= 0.0 && prev_uv.y >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y <= 1.0 {
             let idx0 = u32(p0.y) * params.width + u32(p0.x);
             let idx1 = u32(p1.y) * params.width + u32(p1.x);
             let idx2 = u32(p2.y) * params.width + u32(p2.x);
             let idx3 = u32(p3.y) * params.width + u32(p3.x);

             var c0 = vec3f(0.0);
             var c1 = vec3f(0.0);
             var c2 = vec3f(0.0);
             var c3 = vec3f(0.0);

             // History is already Tonemapped?
             // NO! The accumulation buffer stores the RESULT of the previous frame.
             // If we Inverse Tonemap at the end of this shader, the history (read from accumulation) is linear HDR.
             // So we must Tonemap the history sample too.
             
             if p0.x >= 0 && p0.y >= 0 && p0.x < i32(params.width) && p0.y < i32(params.height) { c0 = resolve_tonemap(history[idx0].rgb); }
             if p1.x >= 0 && p1.y >= 0 && p1.x < i32(params.width) && p1.y < i32(params.height) { c1 = resolve_tonemap(history[idx1].rgb); }
             if p2.x >= 0 && p2.y >= 0 && p2.x < i32(params.width) && p2.y < i32(params.height) { c2 = resolve_tonemap(history[idx2].rgb); }
             if p3.x >= 0 && p3.y >= 0 && p3.x < i32(params.width) && p3.y < i32(params.height) { c3 = resolve_tonemap(history[idx3].rgb); }

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

        // --- Dynamic Blend Factor based on Motion ---
        // Calculate velocity in pixels
        let motion_px = structure_motion * vec2f(f32(params.width), f32(params.height));
        let speed = length(motion_px);

        // If speed is low, we trust history more (High feedback).
        // If speed is high, we trust current frame more (Low feedback).
        // Base feedback: 0.90. Max feedback: 0.97 (or even 0.98 for very static).
        var dynamic_feedback = mix(0.98, 0.85, smoothstep(0.0, 2.0, speed));
        
        // --- Relaxed Clipping for Static Scenes ---
        // If motion is very low, the "Current Neighborhood" might be dominated by noise.
        // The "Clamped History" will thus jitter.
        // To stabilize, we allow some "Unclipped History" to bleed through if we are confident the pixel hasn't moved.
        // Be careful: This causes ghosting if lighting changes rapidly on static objects.
        // But for "Static Image Quality", it's a good tradeoff.
        let static_factor = 1.0 - smoothstep(0.0, 0.5, speed);
        let final_history_color = mix(clamped_history, history_color, static_factor * 0.2); // 20% relax max

        final_tm = mix(tm_filtered, final_history_color, dynamic_feedback);
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
