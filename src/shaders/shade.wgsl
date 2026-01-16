@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var out_color: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_color);
    if id.x >= size.x || id.y >= size.y { return; }

    // G-Buffer 読み込み
    let coord = vec2<i32>(id.xy);
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    let normal_w = textureLoad(gbuffer_normal, coord, 0);
    let albedo = textureLoad(gbuffer_albedo, coord, 0);

    let pos = pos_w.xyz;
    let normal = normal_w.xyz;
    
    // 背景判定 (w < 0.0 means miss/background from gbuffer.wgsl)
    if pos_w.w < 0.0 {
        textureStore(out_color, coord, vec4f(0.0, 0.0, 0.0, 1.0)); // 背景色 (Black for now)
        return;
    }

    // 簡易ライティング (Point Light 1つでテスト)
    let light_pos = vec3f(0.0, 1.9, 0.0);
    let L_vec = light_pos - pos;
    let dist = length(L_vec);
    let L = normalize(L_vec);
    let NdotL = max(dot(normal, L), 0.0);
    
    // Simple falloff or just NdotL
    // User sample: `let color = albedo.rgb * NdotL; // 影なし`

    let color = albedo.rgb * NdotL;

    textureStore(out_color, coord, vec4f(color, 1.0));
}
