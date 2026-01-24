@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var pos_tex: texture_2d<f32>;

struct PostParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
};
@group(0) @binding(5) var<uniform> params: PostParams;

fn gauss(x: f32, sigma: f32) -> f32 {
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
    let sigma_spatial = 2.0;
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

    // Accumulation logic
    var current_acc = vec4f(0.0);
    if params.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(filtered_color * f32(params.spp), f32(params.spp));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;
    
    // Gamma correction (Manual)
    final_color = pow(final_color, vec3f(1.0 / 2.2));

    textureStore(out_tex, vec2i(coords), vec4f(final_color, 1.0));
}
