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


// --- メインの計算関数 ---
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

        let hit = rayQueryGetCommittedIntersection(&rq);

        if hit.kind == 0u {
            break; // 背景は黒
        }

        // 1. マテリアル & メッシュID取得
        let raw_id = hit.instance_custom_data;
        let mesh_id = raw_id >> 16u; // High 16 bits = Mesh ID
        let mat_id = raw_id & 0xFFFFu; // Low 16 bits = Material ID
        let mat = materials[mat_id];

        // 2. 頂点データ補間 (Normal, UV)
        let mesh_info = mesh_infos[mesh_id];
        let idx_offset = mesh_info.index_offset + hit.primitive_index * 3u;
        let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
        let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
        let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        let u_bary = hit.barycentrics.x;
        let v_bary = hit.barycentrics.y;
        let w_bary = 1.0 - u_bary - v_bary;

        let local_normal = normalize(v0.normal.xyz * w_bary + v1.normal.xyz * u_bary + v2.normal.xyz * v_bary);
        let uv = v0.uv.xy * w_bary + v1.uv.xy * u_bary + v2.uv.xy * v_bary;

        // Correct Normal Transformation
        let w2o = hit.world_to_object;
        let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
        let world_normal = normalize(local_normal * m_inv);

        // テクスチャサンプリング (もし mat.tex_id が 0 以上ならサンプリングして base_color に乗算)
        var tex_color = vec4f(1.0);
        // tex_id = 0 はデフォルト（白orチェッカー）として常にサンプリングする
        tex_color = textureSampleLevel(textures, tex_sampler, uv, i32(mat.tex_id), 0.0);

        let final_base_color = mat.base_color * tex_color;

        // 3. エミッション加算 & ライト処理
        let is_front_face = hit.front_face;
        let ffnormal = select(-world_normal, world_normal, is_front_face);


        if !previous_was_diffuse && mat.emission.a > 0.0 && is_front_face {
            accumulated_color += mat.emission.rgb * mat.emission.a * throughput;
        }

        // --- Light Detection ---
        // Strengthが大きいものだけ光源として扱う
        if mat.emission.a > 1.0 {
            break;
        }
        previous_was_diffuse = false;

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
            throughput *= final_base_color.rgb;
        } else if mat.ior > 1.01 || mat.ior < 0.99 { // Dielectric (Glass)
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
            throughput *= final_base_color.rgb;
        } else { // Lambertian (Default)
            // NEE
            if camera.num_lights > 0u {
            
                // 1. ランダムにライトを1つ選ぶ
                let num_lights_f = f32(camera.num_lights);
                let light_idx = u32(rand() * num_lights_f); 
                // 選ばれる確率 (1/N)
                let light_pick_pdf = 1.0 / num_lights_f;

                // 2. そのライト上の点をサンプリング
                let ls = sample_light(light_idx); 

                // 3. シャドウレイの方向と距離
                let to_light = ls.pos - hit_pos;
                let dist_sq = dot(to_light, to_light);
                let dist = sqrt(dist_sq);
                let L = to_light / dist; // ライトへの方向ベクトル

                // 4. Cos項 (N dot L)と(L dot N_light)
                let n_dot_l = max(dot(world_normal, L), 0.0);
                let l_dot_n = max(dot(-L, ls.normal), 0.0); // ライトもこっちを向いているか？

                if n_dot_l > 0.0 && l_dot_n > 0.0 {
                    // 5. 遮蔽テスト (Shadow Ray)
                    // ライトまでの距離(dist)より少し手前(T_MAX)まで飛ばす
                    let offset_pos = hit_pos + world_normal * 0.001;
                    var shadow_rq: ray_query;
                    let flag = 0x4u;
                    rayQueryInitialize(&shadow_rq, tlas, RayDesc(flag, 0xFFu, 0.001, dist - 0.001, offset_pos, L));
                    rayQueryProceed(&shadow_rq);

                    if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u { // 遮蔽なし
                    // 6. 寄与の計算 (Radiance Estimate)
                    // PDF変換: Area Measure -> Solid Angle Measure
                    // pdf_solid = pdf_area * (dist^2 / cos_theta_light)
                        let pdf_solid = ls.pdf * (dist_sq / l_dot_n);
                        let weight = 1.0 / (light_pick_pdf * pdf_solid); // MISウェイト（今回はNEEのみなので単純な逆数）

                        let brdf = final_base_color.rgb / PI; // Lambert Diffuse


                        accumulated_color += ls.emission.rgb * ls.emission.a * brdf * n_dot_l * weight * throughput;
                    }
                }
            }
            previous_was_diffuse = true;

            scatter_dir = ffnormal + random_unit_vector();
            if length(scatter_dir) < 0.001 {
                scatter_dir = ffnormal;
            }
            scatter_dir = normalize(scatter_dir);
            throughput *= final_base_color.rgb;
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