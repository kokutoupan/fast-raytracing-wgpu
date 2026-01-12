@group(0) @binding(0) var raw_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
}

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let uv = vec2f(f32((idx << 1u) & 2u), f32(idx & 2u));
    return VSOut(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let coords = vec2u(in.uv * vec2f(f32(params.width), f32(params.height)));
    let idx = coords.y * params.width + coords.x;

    let raw_color = textureLoad(raw_tex, coords, 0);

    var current_acc = vec4f(0.0);
    if params.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(raw_color.rgb * f32(params.spp), f32(params.spp));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;
    
    // // Gamma correction
    // final_color = pow(final_color, vec3f(1.0 / 2.2));

    return vec4f(final_color, 1.0);
}
