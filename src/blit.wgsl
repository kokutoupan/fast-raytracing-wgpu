@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct BlitParams {
    scale: vec2f,
    _padding: vec2f,
};
@group(0) @binding(2) var<uniform> blit_params: BlitParams;

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
    return VSOut(vec4f(pos[idx] * blit_params.scale, 0.0, 1.0), uv[idx]);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return textureSample(t, s, in.uv);
}
