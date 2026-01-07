enable wgpu_ray_query;

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_tex);
    if id.x >= size.x || id.y >= size.y { return; }

    let uv = (vec2f(id.xy) / vec2f(size)) * 2.0 - 1.0;
    let ray_ori = vec3f(uv.x, -uv.y, 2.0);
    let ray_dir = vec3f(0.0, 0.0, -1.0);

    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, ray_ori, ray_dir));
    rayQueryProceed(&rq);

    let hit = rayQueryGetCommittedIntersection(&rq);
    var color = vec4f(0.1, 0.2, 0.3, 1.0); // 背景色
    if hit.kind != 0u {
        color = vec4f(1.0, 0.6, 0.0, 1.0); // 三角形の色 (オレンジ)
    }

    textureStore(out_tex, vec2<i32>(id.xy), color);
}
