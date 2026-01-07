enable wgpu_ray_query;

// --- 構造体定義 ---
struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    frame_count: u32,
}

struct Material {
    color: vec4f,
    emission: vec4f,
    extra: vec4f, // x: type, y: fuzz, z: ior, w: padding
}

// --- バインドグループ ---
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read_write> accumulation: array<vec4f>;

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

// --- メインの計算関数 ---
fn ray_color(r_in: Ray) -> vec3f {
    var r = r_in;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);

    let MAX_DEPTH = 8u; // 反射回数を増やす (ガラス用に)

    for (var i = 0u; i < MAX_DEPTH; i++) {
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, r.origin, r.dir));
        rayQueryProceed(&rq);

        let hit = rayQueryGetCommittedIntersection(&rq);

        if hit.kind == 0u {
            break; // 背景は黒
        }

        // 1. マテリアル & ジオメトリ情報取得
        let raw_id = hit.instance_custom_data;
        let geo_type = raw_id >> 16u; // High 16 bits = Geometry Type
        let mat_id = raw_id & 0xFFFFu; // Low 16 bits = Material ID
        let mat = materials[mat_id];

        // 2. エミッション加算
        accumulated_color += mat.emission.rgb * throughput;

        // 3. 法線計算
        var local_normal = vec3f(0.0, 1.0, 0.0);
        let tri = hit.primitive_index;
        
        // Geometry Type による分岐
        // 0: Plane (Y-up Quad)
        // 1: Cube
        if geo_type == 1u { // Cube
            // ヒット位置をローカル空間に変換
            // hit.world_to_object は 4x3 行列。
            let w2o = hit.world_to_object;
            let hp = r.origin + r.dir * hit.t;
            let m0 = vec4f(w2o[0], 0.0);
            let m1 = vec4f(w2o[1], 0.0);
            let m2 = vec4f(w2o[2], 0.0);
            let m3 = vec4f(w2o[3], 1.0);
            
            // 行列を手動で構成して乗算 (WGSLの行列コンストラクタは列優先)
            let transform = mat4x4f(m0, m1, m2, m3);
            let local_pos = (transform * vec4f(hp, 1.0)).xyz;

            let abs_p = abs(local_pos);
            if abs_p.x > abs_p.y && abs_p.x > abs_p.z {
                local_normal = vec3f(sign(local_pos.x), 0.0, 0.0);
            } else if abs_p.y > abs_p.z {
                local_normal = vec3f(0.0, sign(local_pos.y), 0.0);
            } else {
                local_normal = vec3f(0.0, 0.0, sign(local_pos.z));
            }
        }
        // Plane (0) defaults to (0, 1, 0) which is correct for Y-up quad in local space

        // Correct Normal Transformation: Transpose(Inverse(Model)) * Normal
        // hit.world_to_object is the Inverse(Model) matrix.
        // In WGSL, vector * matrix multiplication (v * M) is equivalent to M^T * v.
        // Therefore, local_normal * mat3x3(world_to_object) computes Transpose(WorldToObject) * local_normal.
        let w2o = hit.world_to_object;
        let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
        let world_normal = normalize(local_normal * m_inv);

        // DEBUG: Visualize Normals
        // return world_normal * 0.5 + 0.5;

        // ヒット位置
        let hit_pos = r.origin + r.dir * hit.t;

        // 4. マテリアル散乱処理
        let mat_type = u32(mat.extra.x);
        var scatter_dir = vec3f(0.0);
        var absorbed = false;

        if mat_type == 1u { // Metal
            let reflected = reflect(r.dir, world_normal);
            let fuzz = mat.extra.y;
            scatter_dir = reflected + random_unit_vector() * fuzz;
            // 法線と同じ向きなら有効
            if dot(scatter_dir, world_normal) < -0.0001 {
                absorbed = true;
            }
            throughput *= mat.color.rgb; // 金属色
        } else if mat_type == 2u { // Dielectric (Glass)
            let ir = mat.extra.z; // IOR
            let front_face = dot(r.dir, world_normal) < 0.0;
            let normal = select(-world_normal, world_normal, front_face);
            let refraction_ratio = select(ir, 1.0 / ir, front_face);

            let unit_dir = normalize(r.dir);
            let cos_theta = min(dot(-unit_dir, normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            let cannot_refract = refraction_ratio * sin_theta > 1.0;
            var direction = vec3f(0.0);

            if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand() {
                direction = reflect(unit_dir, normal);
            } else {
                direction = refract(unit_dir, normal, refraction_ratio);
            }

            scatter_dir = direction;
            throughput *= mat.color.rgb; // ガラスも色付き可能
        } else { // Lambertian (Default)
            scatter_dir = world_normal + random_unit_vector();
            // 縮退防止
            if length(scatter_dir) < 0.001 {
                scatter_dir = world_normal;
            }
            scatter_dir = normalize(scatter_dir);
            throughput *= mat.color.rgb;
        }

        // 吸収またはロシアンルーレットで打ち切り
        if absorbed || dot(throughput, throughput) < 0.01 {
            break;
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
    let samples_per_frame = 2u; // 高画質化のためサンプル数維持

    for (var s = 0u; s < samples_per_frame; s++) {
        let jitter = vec2f(rand(), rand());
        let uv_jittered = (vec2f(id.xy) + jitter) / vec2f(size);
        let ndc_jittered = vec2f(uv_jittered.x * 2.0 - 1.0, uv_jittered.y * 2.0 - 1.0);

        let target_ndc_jittered = vec4f(ndc_jittered, 1.0, 1.0);
        let target_world_jittered = view_inv * proj_inv * target_ndc_jittered;
        let direction_jittered = normalize(target_world_jittered.xyz / target_world_jittered.w - origin);

        let ray = Ray(origin, direction_jittered);
        pixel_color_linear += ray_color(ray);
    }

    let idx = id.y * size.x + id.x;
    var current_acc = vec4f(0.0);
    if camera.frame_count > 0u {
        current_acc = accumulation[idx];
    }

    let new_acc = current_acc + vec4f(pixel_color_linear, f32(samples_per_frame));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;
    final_color = pow(final_color, vec3f(1.0 / 2.2)); // Gamma

    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}