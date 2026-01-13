enable wgpu_ray_query;

// --- 定数オーバーライド ---
// デフォルト値を設定 (Rust側から指定がなければこれが使われる)
override MAX_DEPTH: u32 = 8u;
override SPP: u32 = 2u;

// --- 構造体定義 ---
struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    frame_count: u32,
}

struct Material {
    base_color: vec4f,
    emission: vec4f,
    roughness: f32,
    metallic: f32,
    ior: f32,
    tex_id: u32,
}

struct Vertex {
    pos: vec4f,
    normal: vec4f,
}

struct MeshInfo {
    vertex_offset: u32,
    index_offset: u32,
    pad: vec2u,
}

// --- バインドグループ ---
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> mesh_infos: array<MeshInfo>;

// --- 乱数生成器 (PCG Hash) ---
var<private> rng_seed: u32;

fn init_rng(pos: vec2u, width: u32, frame: u32) {
    rng_seed = pos.x + pos.y * width + frame * 927163u;
    rng_seed = pcg_hash(rng_seed);
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_seed = pcg_hash(rng_seed);
    return f32(rng_seed) / 4294967295.0;
}

fn random_in_unit_sphere() -> vec3f {
    for (var i = 0; i < 10; i++) {
        let p = vec3f(rand(), rand(), rand()) * 2.0 - 1.0;
        if length(p) < 1.0 {
            return p;
        }
    }
    return vec3f(0.0, 1.0, 0.0);
}

fn random_unit_vector() -> vec3f {
    return normalize(random_in_unit_sphere());
}

// --- Helper Functions ---
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Schlick's approximation
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// --- レイ構造体 ---
struct Ray {
    origin: vec3f,
    dir: vec3f,
}

// --- 頂点法線取得・補間 ---
fn get_interpolated_normal(mesh_id: u32, primitive_index: u32, barycentric: vec2f) -> vec3f {
    let mesh_info = mesh_infos[mesh_id];
    
    // Index Bufferから3つの頂点インデックスを取得
    let idx_offset = mesh_info.index_offset + primitive_index * 3u;
    let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
    let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
    let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

    // Vertex Bufferから3つの法線を取得
    let n0 = vertices[i0].normal.xyz;
    let n1 = vertices[i1].normal.xyz;
    let n2 = vertices[i2].normal.xyz;

    // 重心座標補間 (u, v corresponds to i1, i2; w = 1 - u - v corresponds to i0)
    let u = barycentric.x;
    let v = barycentric.y;
    let w = 1.0 - u - v;

    return normalize(n0 * w + n1 * u + n2 * v);
}

// --- メインの計算関数 ---
fn ray_color(r_in: Ray) -> vec3f {
    const T_MIN = 0.0001;
    const T_MAX = 100.0;
    var r = r_in;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);


    for (var i = 0u; i < MAX_DEPTH; i++) {
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, T_MIN, T_MAX, r.origin, r.dir));
        rayQueryProceed(&rq);

        let hit = rayQueryGetCommittedIntersection(&rq);

        if hit.kind == 0u {
            break; // 背景は黒
        }

        // 1. マテリアル & メッシュID取得
        let raw_id = hit.instance_custom_data;
        let mesh_id = raw_id >> 16u; // High 16 bits = Mesh ID
        let mat_id = raw_id & 0xFFFFu; // Low 16 bits = Material ID
        let mat = materials[mat_id];

        // 2. 法線計算 (Vertex Embedded Normals) - Moved UP before emission
        var local_normal = get_interpolated_normal(mesh_id, hit.primitive_index, hit.barycentrics);

        // Correct Normal Transformation: Transpose(Inverse(Model)) * Normal
        let w2o = hit.world_to_object;
        let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
        let world_normal = normalize(local_normal * m_inv);

        // 3. エミッション加算 & ライト処理
        let is_front_face = hit.front_face;
        let ffnormal = select(-world_normal, world_normal, is_front_face);
        
        // --- Light Detection ---
        // Strengthが大きいものだけ光源として扱う
        if mat.emission.a > 5.0 {
            // Front face only
            if is_front_face {
                accumulated_color += mat.emission.rgb * mat.emission.a * throughput;
                break;
            }
        }

        // Standard Emission (Strengthが小さいもの)
        accumulated_color += mat.emission.rgb * mat.emission.a * throughput;

        // ヒット位置
        let hit_pos = r.origin + r.dir * hit.t;

        // 4. マテリアル散乱処理
        var scatter_dir = vec3f(0.0);
        var absorbed = false;

        if mat.metallic > 0.01 { // Metal
            let reflected = reflect(r.dir, ffnormal);
            let fuzz = mat.roughness;
            scatter_dir = reflected + random_unit_vector() * fuzz;

            if dot(scatter_dir, ffnormal) <= 0.0 {
                absorbed = true;
            }
            throughput *= mat.base_color.rgb;
        } else if mat.ior > 1.01 { // Dielectric (Glass)
            let ir = mat.ior; // IOR
            let refraction_ratio = select(ir, 1.0 / ir, is_front_face);

            let unit_dir = normalize(r.dir);
            let cos_theta = min(dot(-unit_dir, ffnormal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            let cannot_refract = refraction_ratio * sin_theta > 1.0;
            var direction = vec3f(0.0);

            if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand() {
                direction = reflect(unit_dir, ffnormal);
            } else {
                direction = refract(unit_dir, ffnormal, refraction_ratio);
            }

            scatter_dir = direction;
            throughput *= mat.base_color.rgb;
        } else { // Lambertian (Default)
            scatter_dir = ffnormal + random_unit_vector();
            if length(scatter_dir) < 0.001 {
                scatter_dir = ffnormal;
            }
            scatter_dir = normalize(scatter_dir);
            throughput *= mat.base_color.rgb;
        }

        // 吸収
        if absorbed {
            break;
        }

        // ロシアンルーレット
        if i > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if p < 0.01 {
                break;
            }
            if rand() > p {
                break;
            }
            throughput /= p;
        }

        r.origin = hit_pos;
        r.dir = scatter_dir;
    }

    return accumulated_color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_tex);
    if id.x >= size.x || id.y >= size.y { return; }

    init_rng(id.xy, size.x, camera.frame_count);

    let uv = vec2f(id.xy) / vec2f(size);
    let view_inv = mat4x4f(camera.view_inverse[0], camera.view_inverse[1], camera.view_inverse[2], camera.view_inverse[3]);
    let proj_inv = mat4x4f(camera.proj_inverse[0], camera.proj_inverse[1], camera.proj_inverse[2], camera.proj_inverse[3]);
    let origin = view_inv[3].xyz;

    var pixel_color_linear = vec3f(0.0);

    for (var s = 0u; s < SPP; s++) {
        let jitter = vec2f(rand(), rand());
        let uv_jittered = (vec2f(id.xy) + jitter) / vec2f(size);
        let ndc_jittered = vec2f(uv_jittered.x * 2.0 - 1.0, 1.0 - uv_jittered.y * 2.0);

        let target_ndc_jittered = vec4f(ndc_jittered, 1.0, 1.0);
        let target_world_jittered = view_inv * proj_inv * target_ndc_jittered;
        let direction_jittered = normalize(target_world_jittered.xyz / target_world_jittered.w - origin);

        let ray = Ray(origin, direction_jittered);
        pixel_color_linear += ray_color(ray);
    }

    // Output raw averaged color for this frame
    let final_color = pixel_color_linear / f32(SPP);
    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}