enable wgpu_ray_query;

// --- Structures (raytrace.wgslと同じ) ---
struct Camera {
    view_proj: array<vec4f, 4>,
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    view_pos: vec4f,
    prev_view_proj: array<vec4f, 4>,
    frame_count: u32,
    num_lights: u32,
}

struct Material {
    base_color: vec4f,
    light_index: i32,
    _p0: u32,
    _p1: u32,
    transmission: f32,
    roughness: f32,
    metallic: f32,
    ior: f32,
    tex_id: u32,
}

struct VertexAttributes {
    normal: vec2f,
    uv: vec2f
}

fn decode_octahedral_normal(e: vec2f) -> vec3f {
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn encode_octahedral_normal(n: vec3f) -> vec2f {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    let res_base = n.xy * (1.0 / max(l1, 1e-6));
    let res = select(vec2f(0.0), res_base, l1 > 0.0);

    if n.z < 0.0 {
        let x = res.x;
        let y = res.y;
        let sign_x = select(-1.0, 1.0, x >= 0.0);
        let sign_y = select(-1.0, 1.0, y >= 0.0);
        return vec2f(
            (1.0 - abs(y)) * sign_x,
            (1.0 - abs(x)) * sign_y
        );
    }
    return res;
}

struct MeshInfo {
    vertex_offset: u32,
    index_offset: u32,
    pad: vec2u
}

// --- Bindings ---
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> materials: array<Material>;
@group(0) @binding(3) var<storage, read> attributes: array<VertexAttributes>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> mesh_infos: array<MeshInfo>;

// Output Textures
@group(0) @binding(6) var out_pos: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var out_normal: texture_storage_2d<rg32float, write>;
@group(0) @binding(8) var out_albedo: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(9) var out_motion: texture_storage_2d<rg32float, write>;

// Group 1: Textures
@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var textures: texture_2d_array<f32>;


@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_pos);
    if id.x >= size.x || id.y >= size.y { return; }

    // 1. Ray Generation
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let view_inv = mat4x4<f32>(camera.view_inverse[0], camera.view_inverse[1], camera.view_inverse[2], camera.view_inverse[3]);
    let proj_inv = mat4x4<f32>(camera.proj_inverse[0], camera.proj_inverse[1], camera.proj_inverse[2], camera.proj_inverse[3]);

    let origin = view_inv[3].xyz;
    let target_pos = view_inv * proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let direction = normalize(target_pos.xyz / target_pos.w - origin);

    // 2. Intersection (Ray Query)
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 1000.0, origin, direction));
    rayQueryProceed(&rq);

    let committed = rayQueryGetCommittedIntersection(&rq);

    if committed.kind == 0u {
        // Miss (Background)
        let coord = vec2<i32>(id.xy);
        textureStore(out_pos, coord, vec4<f32>(0.0, 0.0, 0.0, -1.0)); // w=-1 for miss
        textureStore(out_normal, coord, vec4<f32>(0.0));
        textureStore(out_albedo, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_motion, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // 3. Hit Processing
    //   (raytrace.wgsl の HitInfo 取得ロジックと同じ)
    let raw_id = committed.instance_custom_data;
    let mesh_id = raw_id >> 16u;
    let mat_id = raw_id & 0xFFFFu;

    let mesh_info = mesh_infos[mesh_id];
    let idx_offset = mesh_info.index_offset + committed.primitive_index * 3u;
    let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
    let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
    let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

    let v0 = attributes[i0];
    let v1 = attributes[i1];
    let v2 = attributes[i2];

    let n0 = decode_octahedral_normal(v0.normal);
    let n1 = decode_octahedral_normal(v1.normal);
    let n2 = decode_octahedral_normal(v2.normal);

    let u_bary = committed.barycentrics.x;
    let v_bary = committed.barycentrics.y;
    let w_bary = 1.0 - u_bary - v_bary;

    let local_normal = normalize(n0 * w_bary + n1 * u_bary + n2 * v_bary);

    let w2o = committed.world_to_object;
    let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
    let normal = normalize(local_normal * m_inv);
    let ffnormal = select(-normal, normal, committed.front_face);

    let pos = origin + direction * committed.t;
    let mat = materials[mat_id];

    let tex_uv = v0.uv.xy * w_bary + v1.uv.xy * u_bary + v2.uv.xy * v_bary;

    let tex_color = textureSampleLevel(textures, tex_sampler, tex_uv, i32(mat.tex_id), 0.0);
    let base_color = mat.base_color.rgb * tex_color.rgb;

    // --- Motion Vector Calculation ---
    // Clip Space: -1..1
    // UV Space: 0..1
    let view_proj = mat4x4<f32>(camera.view_proj[0], camera.view_proj[1], camera.view_proj[2], camera.view_proj[3]);
    let prev_view_proj = mat4x4<f32>(camera.prev_view_proj[0], camera.prev_view_proj[1], camera.prev_view_proj[2], camera.prev_view_proj[3]);

    let curr_clip = view_proj * vec4<f32>(pos, 1.0);
    let prev_clip = prev_view_proj * vec4<f32>(pos, 1.0);

    let curr_ndc = curr_clip.xy / curr_clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;

    let curr_uv = curr_ndc * vec2f(0.5, -0.5) + 0.5; // Y flip for UV
    let prev_uv = prev_ndc * vec2f(0.5, -0.5) + 0.5;

    let motion = prev_uv - curr_uv;

    // 4. Store G-Buffer
    // Pos.w に Material ID を入れる
    let coord = vec2<i32>(id.xy);
    textureStore(out_pos, coord, vec4f(pos, f32(mat_id)));

    let encoded_n = encode_octahedral_normal(ffnormal);
    textureStore(out_normal, coord, vec4f(encoded_n, 0.0, 0.0));

    textureStore(out_albedo, coord, vec4f(base_color, 0.0));
    // Motion
    textureStore(out_motion, coord, vec4f(motion, 0.0, 0.0));
}