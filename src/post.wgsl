@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;

struct PostParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
};
@group(0) @binding(3) var<uniform> params: PostParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    let coords = id.xy;
    let idx = coords.y * params.width + coords.x;

    let raw_color = textureLoad(raw_tex, coords, 0);

    var current_acc = vec4f(0.0);
    if params.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(raw_color.rgb * f32(params.spp), f32(params.spp));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;
    
    // Gamma correction (Manual)
    final_color = pow(final_color, vec3f(1.0 / 2.2));

    textureStore(out_tex, vec2i(coords), vec4f(final_color, 1.0));
}
