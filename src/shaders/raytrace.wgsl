enable wgpu_ray_query;

// --- 定数オーバーライド ---
// デフォルト値を設定 (Rust側から指定がなければこれが使われる)
override MAX_DEPTH: u32 = 8u;
override SPP: u32 = 2u;

// 定数
const PI = 3.14159265359;

// --- 構造体定義 ---
struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    frame_count: u32,
    num_lights: u32,
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
    uv: vec4f,
}

struct MeshInfo {
    vertex_offset: u32,
    index_offset: u32,
    pad: vec2u,
}

// 構造体定義に追加
struct Light {
    position: vec3f,
    type_: u32,
    u: vec3f,
    area: f32,
    v: vec3f,
    pad: u32,
    emission: vec4f,
}

struct LightSample {
    pos: vec3f,     // ライト上のサンプリング点
    normal: vec3f,  // その点の法線（ライトの向き）
    pdf: f32,       // 確率密度（面積測度）
    emission: vec4f // 発光強度
}

// --- バインドグループ ---
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> mesh_infos: array<MeshInfo>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;

@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var textures: texture_2d_array<f32>;

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

// 単位球面上の点をランダムに返す (一様分布)
// Rejection Sampling (Loop) を廃止して、極座標から直接計算する
fn random_unit_vector() -> vec3f {
    let z = rand() * 2.0 - 1.0; // -1.0 ~ 1.0
    let a = rand() * 2.0 * PI;  // 0.0 ~ 2π
    let r = sqrt(1.0 - z * z);

    let x = r * cos(a);
    let y = r * sin(a);

    return vec3f(x, y, z);
}

// 単位球"内"の点が必要な場合 (ボリュームレンダリング等でなければあまり使わないかも？)
// 表面の点(unit_vector)に、距離の3乗根(体積補正)を掛ける
fn random_in_unit_sphere() -> vec3f {
    let r = pow(rand(), 1.0 / 3.0);
    return random_unit_vector() * r;
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

// --- 光源サンプリング ---
// src/shaders/raytrace.wgsl

fn sample_light(light_idx: u32) -> LightSample {
    let light = lights[light_idx];
    var smp: LightSample;
    smp.emission = light.emission;

    // 乱数を2つ取得 (0.0 ~ 1.0)
    let r1 = rand();
    let r2 = rand();

    if light.type_ == 0u { // Quad (Rectangle)
        // 中心から ±u, ±v の範囲
        // r1, r2 は 0~1 なので、 -1~1 に変換
        let su = r1 * 2.0 - 1.0;
        let sv = r2 * 2.0 - 1.0;

        smp.pos = light.position + light.u * su + light.v * sv;
        
        // 法線は u と v の外積（面の向き）
        // Cornell Boxの天井ライト等は下向きに設置されている前提
        smp.normal = normalize(cross(light.u, light.v));
        smp.pdf = 1.0 / light.area; // 面積測度 (1/A)

    } else { // Sphere
        // 球面を一様にサンプリング
        let z = 1.0 - 2.0 * r1;
        let r_xy = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.0 * PI * r2;
        let x = r_xy * cos(phi);
        let y = r_xy * sin(phi);

        let local_dir = vec3f(x, y, z);
        let radius = light.v.x; // v.x に半径が入っているルール

        smp.pos = light.position + local_dir * radius;
        smp.normal = local_dir; // 球の法線は中心からの方向
        smp.pdf = 1.0 / light.area; // 全面積での確率 (1 / 4πr^2)
    }

    return smp;
}


struct HitInfo {
    pos: vec3f,
    normal: vec3f,
    ffnormal: vec3f,
    uv: vec2f,
    mat_id: u32,
    front_face: bool,
    t: f32,
}

// Material evaluation and NEE functions remain here
struct ScatterResult {
    dir: vec3f,
    throughput_mult: vec3f,
    absorbed: bool,
    was_diffuse: bool,
}

fn evaluate_material(
    r_dir: vec3f,
    hit: HitInfo,
    mat: Material,
    base_color: vec3f
) -> ScatterResult {
    var res: ScatterResult;
    res.absorbed = false;
    res.was_diffuse = false;

    if mat.metallic > 0.01 { // Metal
        let reflected = reflect(r_dir, hit.ffnormal);
        res.dir = reflected + random_unit_vector() * mat.roughness;
        res.absorbed = dot(res.dir, hit.ffnormal) <= 0.0;
        res.throughput_mult = base_color;
    } else if mat.ior > 1.01 || mat.ior < 0.99 { // Glass
        let refraction_ratio = select(mat.ior, 1.0 / mat.ior, hit.front_face);
        let unit_dir = normalize(r_dir);
        let cos_theta = min(dot(-unit_dir, hit.ffnormal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > rand() {
            res.dir = reflect(unit_dir, hit.ffnormal);
        } else {
            res.dir = refract(unit_dir, hit.ffnormal, refraction_ratio);
        }
        res.throughput_mult = base_color;
    } else { // Lambertian
        res.dir = normalize(hit.ffnormal + random_unit_vector());
        if length(res.dir) < 0.001 { res.dir = hit.ffnormal; }
        res.throughput_mult = base_color;
        res.was_diffuse = true;
    }

    return res;
}

fn calculate_nee(pos: vec3f, ffnormal: vec3f, throughput: vec3f, base_color: vec3f) -> vec3f {
    if camera.num_lights == 0u { return vec3f(0.0); }

    let num_f = f32(camera.num_lights);
    let ls = sample_light(u32(rand() * num_f));
    let to_light = ls.pos - pos;
    let dist_sq = dot(to_light, to_light);
    let dist = sqrt(dist_sq);
    let L = to_light / dist;

    let n_dot_l = max(dot(ffnormal, L), 0.0);
    let l_dot_n = max(dot(-L, ls.normal), 0.0);

    if n_dot_l > 0.0 && l_dot_n > 0.0 {
        let offset_pos = pos + ffnormal * 0.001;
        var shadow_rq: ray_query;
        rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, 0.001, dist - 0.001, offset_pos, L));
        rayQueryProceed(&shadow_rq);

        if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
            let pdf_solid = ls.pdf * (dist_sq / l_dot_n);
            let weight = 1.0 / ((1.0 / num_f) * pdf_solid);
            return ls.emission.rgb * ls.emission.a * (base_color / PI) * n_dot_l * weight * throughput;
        }
    }
    return vec3f(0.0);
}

fn ray_color(r_in: Ray) -> vec3f {
    const T_MIN = 0.0001;
    const T_MAX = 100.0;
    var r = r_in;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    var previous_was_diffuse = false;

    for (var i = 0u; i < MAX_DEPTH; i++) {
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, T_MIN, T_MAX, r.origin, r.dir));
        rayQueryProceed(&rq);

        let committed = rayQueryGetCommittedIntersection(&rq);
        if committed.kind == 0u { break; }

        // Inline Hit Info Extraction
        var hit: HitInfo;
        let raw_id = committed.instance_custom_data;
        let mesh_id = raw_id >> 16u;
        hit.mat_id = raw_id & 0xFFFFu;

        let mesh_info = mesh_infos[mesh_id];
        let idx_offset = mesh_info.index_offset + committed.primitive_index * 3u;
        let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
        let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
        let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        let u_bary = committed.barycentrics.x;
        let v_bary = committed.barycentrics.y;
        let w_bary = 1.0 - u_bary - v_bary;

        let local_normal = normalize(v0.normal.xyz * w_bary + v1.normal.xyz * u_bary + v2.normal.xyz * v_bary);
        hit.uv = v0.uv.xy * w_bary + v1.uv.xy * u_bary + v2.uv.xy * v_bary;

        let w2o = committed.world_to_object;
        let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
        hit.normal = normalize(local_normal * m_inv);

        hit.front_face = committed.front_face;
        hit.ffnormal = select(-hit.normal, hit.normal, hit.front_face);
        hit.t = committed.t;
        hit.pos = r.origin + r.dir * hit.t;

        let mat = materials[hit.mat_id];

        // 1. Emission
        if !previous_was_diffuse && mat.emission.a > 0.0 && hit.front_face {
            accumulated_color += mat.emission.rgb * mat.emission.a * throughput;
        }

        if mat.emission.a > 1.0 { break; } // Light source hit

        // 2. Base Color & Texture
        let tex_color = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.tex_id), 0.0);
        let base_color = (mat.base_color * tex_color).rgb;

        // 3. NEE
        if mat.metallic <= 0.01 && (mat.ior <= 1.01 && mat.ior >= 0.99) {
            accumulated_color += calculate_nee(hit.pos, hit.ffnormal, throughput, base_color);
            previous_was_diffuse = true;
        } else {
            previous_was_diffuse = false;
        }

        // 4. Scattering
        let sc = evaluate_material(r.dir, hit, mat, base_color);
        if sc.absorbed { break; }

        throughput *= sc.throughput_mult;
        r.origin = hit.pos;
        r.dir = sc.dir;

        // Russian Roulette
        if i > 3u {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if p < 0.01 || rand() > p { break; }
            throughput /= p;
        }
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
    // // --- DEBUG: 光源数チェック ---
    // // final_color を上書きします

    // let n = camera.num_lights;
    // if n == 0u {
    //     final_color = vec3f(1.0, 0.0, 0.0); // 赤: 0個 (データ来てないかも？)
    // } else if n == 1u {
    //     final_color = vec3f(0.0, 1.0, 0.0); // 緑: 1個 (OK)
    // } else if n == 2u {
    //     final_color = lights[0].emission.rgb; // 青: 2個 (OK)
    // } else {
    //     final_color = vec3f(1.0, 1.0, 0.0); // 黄: 3個以上 (OK)
    // }

    // // --- DEBUG END ---

    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}