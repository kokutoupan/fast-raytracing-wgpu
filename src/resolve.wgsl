@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4f>;

struct ResolveParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
};
@group(0) @binding(2) var<uniform> resolve_params: ResolveParams;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, -1.0)
    );
    var uv = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0),
        vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0)
    );
    return VSOut(vec4f(pos[idx], 0.0, 1.0), uv[idx]);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    // 精度によるはみ出しを防ぐため clamp
    let coords = vec2u(clamp(in.uv, vec2f(0.0), vec2f(0.9999)) * vec2f(f32(resolve_params.width), f32(resolve_params.height)));
    let idx = coords.y * resolve_params.width + coords.x;

    let raw_color = textureLoad(raw_tex, coords, 0);

    var current_acc = vec4f(0.0);
    if resolve_params.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(raw_color.rgb * f32(resolve_params.spp), f32(resolve_params.spp));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;
    
    // Gamma correction
    // final_color = pow(final_color, vec3f(1.0 / 2.2));

    return vec4f(final_color, 1.0);
}
