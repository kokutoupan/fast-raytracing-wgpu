enable wgpu_ray_query;

// --- Structures (raytrace.wgslと同じ) ---
struct Camera {
    view_proj: mat4x4f,
    view_inverse: mat4x4f,
    proj_inverse: mat4x4f,
    view_pos: vec4f,
    prev_view_proj: mat4x4f,
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
    normal_tex_id: u32,
    occlusion_tex_id: u32,
    emissive_tex_id: u32,
}

struct VertexAttributes {
    normal: vec2f,
    uv: vec2f,
    tangent: vec4f,
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
@group(0) @binding(7) var out_normal: texture_storage_2d<rgba32float, write>;
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

    let view_inv = camera.view_inverse;
    let proj_inv = camera.proj_inverse;

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

    let t0 = v0.tangent.xyz;
    let t1 = v1.tangent.xyz;
    let t2 = v2.tangent.xyz;

    let u_bary = committed.barycentrics.x;
    let v_bary = committed.barycentrics.y;
    let w_bary = 1.0 - u_bary - v_bary;

    let local_normal = normalize(n0 * w_bary + n1 * u_bary + n2 * v_bary);
    let local_tangent = normalize(t0 * w_bary + t1 * u_bary + t2 * v_bary);
    let tangent_sign = v0.tangent.w; // Assume constant sign across triangle

    let w2o = committed.world_to_object;
    let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
    
    // Transform Normal and Tangent to World Space
    let normal_w = normalize(local_normal * m_inv);
    let tangent_w = normalize(local_tangent * m_inv);
    let bitangent_w = cross(normal_w, tangent_w) * tangent_sign;

    // Gram-Schmidt / TBN Matrix
    let T = normalize(tangent_w - normal_w * dot(normal_w, tangent_w));
    let B = normalize(cross(normal_w, T)) * tangent_sign; // Or use bitangent_w directly
    let N = normal_w;
    let TBN = mat3x3f(T, B, N);

    let ffnormal = select(-normal_w, normal_w, committed.front_face);

    let pos = origin + direction * committed.t;
    let mat = materials[mat_id];

    let tex_uv = v0.uv.xy * w_bary + v1.uv.xy * u_bary + v2.uv.xy * v_bary;

    // --- Texture Sampling ---
    // 1. Base Color
    var tex_color = vec4f(1.0);
    if mat.tex_id != 4294967295u {
        tex_color = textureSampleLevel(textures, tex_sampler, tex_uv, i32(mat.tex_id), 0.0);
    }

    // 2. Occlusion
    var occlusion = 1.0;
    if mat.occlusion_tex_id != 4294967295u {
        occlusion = textureSampleLevel(textures, tex_sampler, tex_uv, i32(mat.occlusion_tex_id), 0.0).r;
    }

    // 4. Normal Map
    var normal_local = vec3f(0.0, 0.0, 1.0);
    if mat.normal_tex_id != 4294967295u {
        let normal_map = textureSampleLevel(textures, tex_sampler, tex_uv, i32(mat.normal_tex_id), 0.0).rgb;
        normal_local = normalize(normal_map * 2.0 - 1.0);
    }
    
    // Perturb Normal
    var final_normal = ffnormal;
    if mat.normal_tex_id != 4294967295u {
        let tangent_sign = v0.tangent.w;
        // ... (TBN calc) ...
        // We use v0.tangent here, but we should probably interpolate tangents?
        // Wait, I calculated tangent_w above!
        
        let N_ff = ffnormal; 
        // Re-orthogonalize T against N_ff
        let T_ff = normalize(tangent_w - N_ff * dot(N_ff, tangent_w));
        let B_ff = normalize(cross(N_ff, T_ff)) * tangent_sign;
        let TBN_ff = mat3x3f(T_ff, B_ff, N_ff);
        
        final_normal = normalize(TBN_ff * normal_local);
    }

    let base_color = mat.base_color.rgb * tex_color.rgb * occlusion; // Apply Occlusion to Albedo? Or separate channel?
    // Common GLTF: Occlusion is separate. But for simple G-Buffer we might bake it.
    // Actually, storing Occlusion in G-Buffer would be nice for separate AO pass, but we are doing path tracing.
    // "Pre-baked AO" in Path Tracing is kind of "cheat" but helps with detail.
    // Let's burn it into Albedo for now or just multiply.

    // --- Motion Vector Calculation ---
    // Clip Space: -1..1
    // UV Space: 0..1
    let view_proj = camera.view_proj;
    let prev_view_proj = camera.prev_view_proj;

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

    let encoded_n = encode_octahedral_normal(final_normal);
    textureStore(out_normal, coord, vec4f(encoded_n, tex_uv));

    textureStore(out_albedo, coord, vec4f(base_color, 1.0)); // Occlusion is burned in
    // Motion
    textureStore(out_motion, coord, vec4f(motion, 0.0, 0.0));
}