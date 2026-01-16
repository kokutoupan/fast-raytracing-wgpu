enable wgpu_ray_query;

// --- 定数オーバーライド ---
// デフォルト値を設定 (Rust側から指定がなければこれが使われる)
override MAX_DEPTH: u32 = 8u;
override SPP: u32 = 2u;

// 定数
const PI = 3.14159265359;

// --- 構造体定義 ---
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
    _p2: u32,
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

struct Reservoir {
    y: u32,       // 選ばれたライトID
    w_sum: f32,   // 重みの合計
    M: u32,       // 処理した候補数
    W: f32,       // 最終的なMISウェイト
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
@group(0) @binding(8) var<storage, read_write> reservoirs_in: array<Reservoir>;
@group(0) @binding(9) var<storage, read_write> reservoirs_out: array<Reservoir>;

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

fn init_reservoir() -> Reservoir {
    return Reservoir(0u, 0.0, 0u, 0.0);
}

fn update_reservoir(r: ptr<function, Reservoir>, y_new: u32, w_new: f32) {
    (*r).w_sum += w_new;
    (*r).M += 1u;
    if rand() * (*r).w_sum < w_new {
        (*r).y = y_new;
    }
}

// ターゲット評価関数: 「影を無視して、このライトがどれくらい明るそうか」
fn evaluate_target(light_id: u32, pos: vec3f, normal: vec3f) -> f32 {
    let light = lights[light_id];
    let le = length(light.emission.rgb) * light.emission.a;

    let to_light = light.position - pos;
    let dist_sq = dot(to_light, to_light);
    
    // 簡易的な幾何項 (cosθ / dist^2)
    // ※選抜段階では「点光源」扱いで高速化します
    let dist = sqrt(dist_sq);
    let L = to_light / dist;
    let n_dot_l = max(dot(normal, L), 0.0);

    // 面積を考慮 (Areaが大きいライトほど選ばれやすくする)
    return (le * n_dot_l * light.area) / max(0.001, dist_sq);
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

// =================================================================
//   MATH & GGX HELPER FUNCTIONS
// =================================================================

// 正規直交基底を作る (法線 n を Z軸とする座標系)
fn make_orthonormal_basis(n: vec3f) -> mat3x3f {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3f(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3f(b, sign + n.y * n.y * a, -n.y);
    return mat3x3f(tangent, bitangent, n);
}

// Fresnel (Schlick)
fn fresnel_schlick(f0: vec3f, v_dot_h: f32) -> vec3f {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// NDF (GGX / Trowbridge-Reitz)
fn ndf_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Geometry (Smith / Schlick-GGX)
fn geometry_schlick_ggx(n_dot_v: f32, k: f32) -> f32 {
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    // Path Tracing (sample_bsdf) では k = alpha^2 / 2 を使用
    let r = roughness;
    let k = (r * r) / 2.0;
    let ggx1 = geometry_schlick_ggx(n_dot_l, k);
    let ggx2 = geometry_schlick_ggx(n_dot_v, k);
    return ggx1 * ggx2;
}

// GGX VNDF Sampling (Visible Normal Distribution Function)
// 視線方向 wo から見えるマイクロファセットの法線 wm をサンプリングする
fn sample_ggx_vndf(wo: vec3f, roughness: f32, u: vec2f) -> vec3f {
    let alpha = roughness * roughness;

    // View vector to hemisphere configuration
    let Vh = normalize(vec3f(alpha * wo.x, alpha * wo.y, wo.z));

    // Orthonormal basis
    let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    let T1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-Vh.y, Vh.x, 0.0) * inverseSqrt(lensq), lensq > 0.0);
    let T2 = cross(Vh, T1);

    // Parameterization
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let t1 = r * cos(phi);
    let t2 = r * sin(phi);
    let s = 0.5 * (1.0 + Vh.z);
    let t2_lerp = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Reprojection
    let Nh = t1 * T1 + t2_lerp * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2_lerp * t2_lerp)) * Vh;

    // Back to ellipsoid configuration
    let Ne = normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

// =================================================================
//   BSDF & EVALUATION FUNCTIONS
// =================================================================

struct BsdfSample {
    wi: vec3f,          // 次のレイの方向 (World Space)
    pdf: f32,           // 確率密度 (Solid Angle)
    weight: vec3f,      // throughput への寄与 (f * cos / pdf)
    is_delta: bool,     // 鏡面反射など (NEE不可) かどうか
}

// 確率密度関数 (PDF) の評価
fn eval_pdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material) -> f32 {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    // Metal (GGX)
    if mat.metallic > 0.01 {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return 0.0; }

        let h = normalize(wi + wo);
        let n_dot_h = max(dot(normal, h), 0.0);
        let h_dot_v = max(dot(h, wo), 0.0);

        let d = ndf_ggx(n_dot_h, mat.roughness);
        // VNDF sampling PDF: p_wi = D * G1 / (4 * n.v)
        // ※ここでは一般的な PDF = (D * n.h) / (4 * h.v) ではなく、VNDFのPDFを使います
        let k = (mat.roughness * mat.roughness) / 2.0;
        let g1 = geometry_schlick_ggx(n_dot_v, k);
        return (d * g1) / (4.0 * n_dot_v);
    }

    // Glass / Specular (Delta) -> PDF is 0 for analytical evaluation
    if mat.ior > 1.01 || mat.ior < 0.99 {
        return 0.0;
    }

    // Lambert
    return max(n_dot_l, 0.0) / PI;
}

// BSDF (BRDF) の評価値 f を返す (cos項は含まない定義)
fn eval_bsdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material, base_color: vec3f) -> vec3f {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    // Metal (GGX)
    if mat.metallic > 0.01 {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return vec3f(0.0); }

        let h = normalize(wi + wo);
        let n_dot_h = max(dot(normal, h), 0.0);
        let h_dot_v = max(dot(h, wo), 0.0);

        let D = ndf_ggx(n_dot_h, mat.roughness);
        let G = geometry_smith(n_dot_l, n_dot_v, mat.roughness);
        let F = fresnel_schlick(base_color, h_dot_v); // Metal color is F0

        // Cook-Torrance Specular BRDF: f = (D * F * G) / (4 * n.l * n.v)
        let numerator = D * G * F;
        let denominator = 4.0 * n_dot_l * n_dot_v;
        return numerator / max(denominator, 0.001);
    }

    // Glass (Delta) -> Evaluate is 0
    if mat.ior > 1.01 || mat.ior < 0.99 {
        return vec3f(0.0);
    }

    // Lambert: f = c / PI
    return base_color / PI;
}

fn sample_bsdf(
    wo: vec3f,       // 視線方向 (-ray.dir)
    hit: HitInfo,    // 衝突点情報
    mat: Material,   // マテリアル
    base_color: vec3f // テクスチャ適用後の色
) -> BsdfSample {
    var smp: BsdfSample;
    smp.is_delta = false;

    // --- Metal (GGX) ---
    if mat.metallic > 0.01 {
        // 1. 接空間へ変換
        let tbn = make_orthonormal_basis(hit.ffnormal);
        let wo_local = transpose(tbn) * wo; 
        
        // 2. VNDFサンプリング (法線を決める)
        let r = vec2f(rand(), rand());
        let wm_local = sample_ggx_vndf(wo_local, mat.roughness, r);
        let wm = tbn * wm_local; 

        // 3. 反射方向を決める
        smp.wi = reflect(-wo, wm);
        
        // 幾何的なチェック
        let n_dot_l = dot(hit.ffnormal, smp.wi);
        let n_dot_v = dot(hit.ffnormal, wo);

        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            smp.weight = vec3f(0.0);
            smp.pdf = 0.0;
            return smp;
        }

        smp.is_delta = false; // GGXはNEE可能（今回はOFFにしますが）
        smp.pdf = eval_pdf(hit.ffnormal, smp.wi, wo, mat);

        // F項 (フレネル: 色)
        let F = fresnel_schlick(base_color, dot(wo, wm));
        
        // G1項 (遮蔽: 明るさ)
        let k = (mat.roughness * mat.roughness) / 2.0;
        let G1_l = geometry_schlick_ggx(n_dot_l, k);

        smp.weight = F * G1_l;

        return smp;
    }

    // --- Glass (Delta) ---
    if mat.ior > 1.01 || mat.ior < 0.99 {
        smp.is_delta = true;
        smp.pdf = 0.0;

        let refraction_ratio = select(mat.ior, 1.0 / mat.ior, hit.front_face);
        let cos_theta = min(dot(wo, hit.ffnormal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > rand() {
            smp.wi = reflect(-wo, hit.ffnormal);
        } else {
            smp.wi = refract(-wo, hit.ffnormal, refraction_ratio);
        }
        smp.weight = base_color;
        return smp;
    }

    // --- Lambert (Diffuse) ---
    smp.wi = normalize(hit.ffnormal + random_unit_vector());
    if length(smp.wi) < 0.001 { smp.wi = hit.ffnormal; }

    smp.is_delta = false;
    let n_dot_l = max(dot(hit.ffnormal, smp.wi), 0.0);
    smp.pdf = n_dot_l / PI;

    if smp.pdf > 0.0 {
        smp.weight = base_color;
    } else {
        smp.weight = vec3f(0.0);
    }

    return smp;
}

// =================================================================
//   NEE FUNCTION (UPDATED)
// =================================================================

fn run_ris_candidate_search(pos: vec3f, ffnormal: vec3f, num_lights: u32) -> Reservoir {
    const CANDIDATE_COUNT = 16u;
    var r = init_reservoir();

    for (var i = 0u; i < CANDIDATE_COUNT; i++) {
        // A. ランダムにライトを選ぶ (Source PDF = 1/N)
        let light_idx = u32(rand() * f32(num_lights));
        if light_idx >= num_lights { continue; }

        // B. 重要度を計算 (Target PDF = p_hat)
        let p_hat = evaluate_target(light_idx, pos, ffnormal);

        // C. ウェイト w = p_hat / p_source (p_source=1/N は定数なので省略可)
        let w = p_hat;

        update_reservoir(&r, light_idx, w);
    }
    return r;
}

fn apply_temporal_reuse(r: ptr<function, Reservoir>, pos: vec3f, ffnormal: vec3f, pixel_idx: vec2u) {
    let size = textureDimensions(out_tex);
    let prev_r = reservoirs_in[pixel_idx.y * size.x + pixel_idx.x];

    let p_hat_prev = evaluate_target(prev_r.y, pos, ffnormal);

    if prev_r.M > 0u && p_hat_prev > 0.0 {
        let limit_M = min(prev_r.M, 20u);
        update_reservoir(r, prev_r.y, p_hat_prev * prev_r.W * f32(limit_M));
        (*r).M += (limit_M - 1u);
    }
}

fn store_reservoir(r_in: Reservoir, pos: vec3f, ffnormal: vec3f, pixel_idx: vec2u) {
    let size = textureDimensions(out_tex);
    var r = r_in;

    let p_hat_curr = evaluate_target(r.y, pos, ffnormal);
    r.W = 0.0;
    if p_hat_curr > 0.0 {
        r.W = (r.w_sum / f32(r.M)) / p_hat_curr;
    }
    reservoirs_out[pixel_idx.y * size.x + pixel_idx.x] = r;
}

fn evaluate_reservoir_lighting(
    r_in: Reservoir,
    pos: vec3f,
    ffnormal: vec3f,
    wo: vec3f,
    throughput: vec3f,
    mat: Material,
    base_color: vec3f,
    num_lights: u32
) -> vec3f {
    var r = r_in;
    // --- Phase 2: Shading (本番計算) ---
    // 選ばれたエース級ライト(r.y)に対してのみ、重いシャドウレイを飛ばす
    let selected_light_id = r.y;
    // W (MIS weight) の計算
    let p_hat_selected = evaluate_target(selected_light_id, pos, ffnormal);
    if p_hat_selected <= 0.0 { return vec3f(0.0); }

    // RIS Weight: W = (1/M) * (sum(w) / p_hat)
    // p_source = 1/N なので、最後に N (= num_f) を掛ける
    r.W = r.w_sum / (f32(r.M) * p_hat_selected) * f32(num_lights);

    let ls = sample_light(selected_light_id);

    let offset_pos = pos + ffnormal * 0.001;
    let to_light = ls.pos - offset_pos;
    let dist_sq = dot(to_light, to_light);
    let dist = sqrt(dist_sq);
    let L = to_light / dist; // wi

    let n_dot_l = max(dot(ffnormal, L), 0.0);
    let l_dot_n = max(dot(-L, ls.normal), 0.0);

    // 表面がライトを向いていて、かつライトの表面が見えている場合
    if n_dot_l > 0.0 && l_dot_n > 0.0 {
        var shadow_rq: ray_query;
        let t_max = max(0.0, dist - 0.001);
        rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, 0.001, t_max, offset_pos, L));
        rayQueryProceed(&shadow_rq);

        if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {

            // BSDF (BRDF) の評価
            let f = eval_bsdf(ffnormal, L, wo, mat, base_color);
            let light_area = 1.0 / ls.pdf;

            // ReSTIR DI Estimator:
            // L = f * Le * G * W
            // G = (cosθ * cosθ') / dist^2
            let G = (n_dot_l * l_dot_n) / max(0.001, dist_sq);

            return ls.emission.rgb * ls.emission.a * f * G * r.W * throughput * light_area;
        }
    }
    return vec3f(0.0);
}

fn calculate_nee(
    pos: vec3f,
    ffnormal: vec3f,
    wo: vec3f,           // 視線方向 (-ray.dir)
    throughput: vec3f,
    mat: Material,       // マテリアル
    base_color: vec3f,
    pixel_idx: vec2u,
    depth: u32,
) -> vec3f {
    if camera.num_lights == 0u { return vec3f(0.0); }

    // --- Phase 1: RIS (候補選抜) ---
    var r = run_ris_candidate_search(pos, ffnormal, camera.num_lights);

    // --- Temporal Reuse ---
    if depth == 0u {
        apply_temporal_reuse(&r, pos, ffnormal, pixel_idx);
        store_reservoir(r, pos, ffnormal, pixel_idx);
    }

    // --- Phase 2: Shading (本番計算) ---
    return evaluate_reservoir_lighting(r, pos, ffnormal, wo, throughput, mat, base_color, camera.num_lights);
}

fn ray_color(r_in: Ray, pixel_idx: vec2u) -> vec3f {
    const T_MIN = 0.0001;
    const T_MAX = 100.0;
    var r = r_in;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    var previous_was_diffuse = false;
    var last_bsdf_pdf: f32 = 0.0;

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

        // 1. Emission (発光の処理)
        if mat.light_index >= 0 && hit.front_face {
            let light = lights[mat.light_index];
            var mis_weight = 1.0;

            if previous_was_diffuse {
            // A. 前回のBSDF確率（持ち越しておいた変数）
                let p_bsdf = last_bsdf_pdf;

            // B. 今回のNEE確率（逆算）
                let dist_sq = hit.t * hit.t;
                let light_cos = max(dot(hit.ffnormal, -r.dir), 0.0);

                let light_area = light.area;
                let num_lights = f32(camera.num_lights);

            // p_nee = (1 / Area) * (dist^2 / cosθ) * (1 / ライト個数)
                var p_nee = (1.0 / light_area) * (dist_sq / light_cos) * (1.0 / num_lights);

                if light_cos < 0.001 { p_nee = 0.0; }

            // バランスヒューリスティック
                mis_weight = p_bsdf / (p_bsdf + p_nee);
            }

            accumulated_color += light.emission.rgb * light.emission.a * throughput * mis_weight;
                break;
        }

        if mat.light_index >= 0 && lights[mat.light_index].emission.a > 1.0 { break; } // Light source hit

        // 2. Base Color & Texture
        let tex_color = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.tex_id), 0.0);
        let base_color = (mat.base_color * tex_color).rgb;

        // 3. NEE
        // 修正: 鏡(Glass) または 金属(Metal) は NEE を行わない
        // 金属は NEE でヒットする確率が低く、BSDFサンプリング(反射)に任せたほうが品質が良いため
        let is_specular_delta = (mat.ior > 1.01 || mat.ior < 0.99) || (mat.metallic > 0.01);

        if !is_specular_delta {
            // Lambert (非金属) のみここで光源を探す
            accumulated_color += calculate_nee(hit.pos, hit.ffnormal, -r.dir, throughput, mat, base_color, pixel_idx, i);
            previous_was_diffuse = true;
        } else {
            // 金属やガラスはここで光源を探さず、次の sample_bsdf でレイを飛ばして直接当てる
            previous_was_diffuse = false;
        }

        // 4. Scattering
        let sc = sample_bsdf(-r.dir, hit, mat, base_color); // wo = -r.dir

        // 吸収判定: ウェイトが真っ黒(0,0,0)なら、これ以上計算しても意味がないので終了
        if sc.weight.x <= 0.0 && sc.weight.y <= 0.0 && sc.weight.z <= 0.0 { break; }

        last_bsdf_pdf = sc.pdf;

        // スループットとレイの更新
        throughput *= sc.weight;

        r.origin = hit.pos;
        r.dir = sc.wi;

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
        pixel_color_linear += ray_color(ray, id.xy);
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