@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let uv = vec2f(f32((idx << 1u) & 2u), f32(idx & 2u));
    return VSOut(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return textureSample(t, s, in.uv);
}
