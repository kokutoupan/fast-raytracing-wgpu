enable wgpu_ray_query;

// --- Constants & Structs ---
const PI = 3.14159265359;
const MAX_DEPTH = 8u;

// --- Structs ---
struct Camera {
    view_proj: mat4x4f,
    view_inv: mat4x4f,
    proj_inv: mat4x4f,
    view_pos: vec4f,
    prev_view_proj: mat4x4f,
    frame_count: u32,
    num_lights: u32,
}

struct Light {
    position: vec3f,
    type_: u32,
    u: vec3f,
    area: f32,
    v: vec3f,
    pad: u32,
    emission: vec4f,
}

struct Reservoir {
    y: u32,
    w_sum: f32,
    M: u32,
    W: f32,
    s_path: vec3f,
    p_hat: f32,
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

struct VertexAttributes {
    normal: vec2f,
    uv: vec2f,
}

struct MeshInfo {
    vertex_offset: u32,
    index_offset: u32,
    pad: vec2u,
}

struct BsdfSample {
    wi: vec3f,
    pdf: f32,
    weight: vec3f,
    is_delta: bool,
}

struct LightSample {
    pos: vec3f,
    normal: vec3f,
    pdf: f32,
    emission: vec4f,
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

struct PathResult {
    radiance: vec3f,
    v1_pos: vec3f,
    v1_normal: vec3f,
    valid_v1: bool,
}

// --- Bindings ---

// Group 0: G-Buffer & Scene context (Lights, Camera)
@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_motion: texture_2d<f32>; // For temporal reuse

@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<uniform> scene_info: vec4u; // x=light_count, y=frame_count

@group(0) @binding(7) var tlas: acceleration_structure;
@group(0) @binding(8) var<storage, read> materials: array<Material>;
@group(0) @binding(9) var<storage, read> attributes: array<VertexAttributes>;
@group(0) @binding(10) var<storage, read> indices: array<u32>;
@group(0) @binding(11) var<storage, read> mesh_infos: array<MeshInfo>;

@group(0) @binding(12) var prev_gbuffer_pos: texture_2d<f32>;
@group(0) @binding(13) var prev_gbuffer_normal: texture_2d<f32>;
@group(0) @binding(14) var out_tex: texture_storage_2d<rgba32float, write>;

// Group 1: Textures
@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var textures: texture_2d_array<f32>;

// Group 2: Reservoirs
@group(2) @binding(0) var<storage, read> in_reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> out_reservoirs: array<Reservoir>;

// --- RNG ---
var<private> rng_seed: u32;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand() -> f32 {
    rng_seed = pcg_hash(rng_seed);
    return f32(rng_seed) / 4294967295.0;
}

fn random_unit_vector() -> vec3f {
    let z = rand() * 2.0 - 1.0;
    let a = rand() * 2.0 * PI;
    let r = sqrt(1.0 - z * z);
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3f(x, y, z);
}

fn rand_lcg(state: ptr<function, u32>) -> f32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

fn decode_octahedral_normal(e: vec2f) -> vec3f {
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// --- Math Helpers ---
fn make_orthonormal_basis(n: vec3f) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3f(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3f(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(tangent, bitangent, n);
}

fn fresnel_schlick(f0: vec3f, v_dot_h: f32) -> vec3f {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// --- Helper Functions ---
fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Schlick's approximation
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn ndf_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn geometry_schlick_ggx(n_dot_v: f32, k: f32) -> f32 {
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    return geometry_schlick_ggx(n_dot_l, k) * geometry_schlick_ggx(n_dot_v, k);
}

fn sample_ggx_vndf(wo: vec3f, roughness: f32, u: vec2f) -> vec3f {
    let alpha = roughness * roughness;
    let Vh = normalize(vec3f(alpha * wo.x, alpha * wo.y, wo.z));
    let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    let T1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-Vh.y, Vh.x, 0.0) * inverseSqrt(lensq), lensq > 0.0);
    let T2 = cross(Vh, T1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let t1 = r * cos(phi);
    let t2 = r * sin(phi);
    let s = 0.5 * (1.0 + Vh.z);
    let t2_lerp = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    let Nh = t1 * T1 + t2_lerp * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2_lerp * t2_lerp)) * Vh;
    return normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

// --- Light Sampling ---
fn sample_light(light_idx: u32) -> LightSample {
    let light = lights[light_idx];
    var smp: LightSample;
    smp.emission = light.emission;

    let r1 = rand();
    let r2 = rand();

    if light.type_ == 0u { // Quad
        let su = r1 * 2.0 - 1.0;
        let sv = r2 * 2.0 - 1.0;
        smp.pos = light.position + light.u * su + light.v * sv;
        smp.normal = normalize(cross(light.u, light.v));
        smp.pdf = 1.0 / light.area;
    } else { // Sphere
        let z = 1.0 - 2.0 * r1;
        let r_xy = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.0 * PI * r2;
        let x = r_xy * cos(phi);
        let y = r_xy * sin(phi);
        let local_dir = vec3f(x, y, z);
        smp.pos = light.position + local_dir * light.v.x;
        smp.normal = local_dir;
        smp.pdf = 1.0 / light.area;
    }
    return smp;
}

// --- BSDF Evaluation ---
// --- BSDF Evaluation ---
fn eval_pdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material, base_color: vec3f) -> f32 {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    // Glass (Delta)
    if mat.ior > 1.01 || mat.ior < 0.99 { return 0.0; }

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return 0.0; }

    // Unified PBR
    let F0 = mix(vec3f(0.04), base_color, mat.metallic);
    let F = fresnel_schlick(F0, max(dot(normal, wo), 0.0));
    let lum_spec = luminance(F);
    let lum_diff = luminance(base_color * (1.0 - mat.metallic));
    let prob_spec = clamp(lum_spec / (lum_spec + lum_diff + 0.001), 0.05, 0.95);

    // Specular PDF
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(normal, h), 0.0);
    let d = ndf_ggx(n_dot_h, mat.roughness);
    let k = (mat.roughness * mat.roughness) / 2.0;
    let g1 = geometry_schlick_ggx(n_dot_v, k);
    let pdf_spec = (d * g1) / (4.0 * n_dot_v);

    // Diffuse PDF
    let pdf_diff = max(n_dot_l, 0.0) / PI;

    return prob_spec * pdf_spec + (1.0 - prob_spec) * pdf_diff;
}

fn eval_bsdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material, base_color: vec3f) -> vec3f {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    // Glass (Delta)
    if mat.ior > 1.01 || mat.ior < 0.99 { return vec3f(0.0); }

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return vec3f(0.0); }

    // Constants
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(normal, h), 0.0);
    let h_dot_v = max(dot(h, wo), 0.0);
    let F0 = mix(vec3f(0.04), base_color, mat.metallic);

    // Specular Term (GGX)
    let D = ndf_ggx(n_dot_h, mat.roughness);
    let G = geometry_smith(n_dot_l, n_dot_v, mat.roughness);
    let F = fresnel_schlick(F0, h_dot_v);
    let specular = (D * G * F) / max(4.0 * n_dot_l * n_dot_v, 0.001);

    // Diffuse Term (Lambert)
    // Metallic surfaces have no diffuse contribution
    let kD = (vec3f(1.0) - F) * (1.0 - mat.metallic);
    let diffuse = kD * base_color / PI;

    return diffuse + specular;
}

fn sample_bsdf(wo: vec3f, hit: HitInfo, mat: Material, base_color: vec3f) -> BsdfSample {
    var smp: BsdfSample;
    smp.is_delta = false;

    // Glass (Delta) - Remains separate
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

    // Unified PBR Stochastic Sampling
    let F0 = mix(vec3f(0.04), base_color, mat.metallic);
    let F_view = fresnel_schlick(F0, max(dot(hit.ffnormal, wo), 0.0));
    
    // Calculate selection probability based on estimated luminance contribution
    let lum_spec = luminance(F_view);
    let lum_diff = luminance(base_color * (1.0 - mat.metallic));
    let prob_spec = clamp(lum_spec / (lum_spec + lum_diff + 0.001), 0.05, 0.95);

    let rnd = rand();
    if rnd < prob_spec {
        // Sample Specular (GGX)
        let tbn = make_orthonormal_basis(hit.ffnormal);
        let wo_local = transpose(tbn) * wo;
        let r_uv = vec2f(rand(), rand());
        let wm_local = sample_ggx_vndf(wo_local, mat.roughness, r_uv);
        let wm = tbn * wm_local;
        smp.wi = reflect(-wo, wm);
    } else {
        // Sample Diffuse (Lambert)
        smp.wi = normalize(hit.ffnormal + random_unit_vector());
    }

    let n_dot_l = dot(hit.ffnormal, smp.wi);
    let n_dot_v = dot(hit.ffnormal, wo);

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        smp.weight = vec3f(0.0);
        smp.pdf = 0.0;
        return smp;
    }

    // Evaluate full BSDF and PDF for the chosen direction
    // Note: We need to pass base_color locally here, so eval_bsdf call is cleaner
    let bsdf_val = eval_bsdf(hit.ffnormal, smp.wi, wo, mat, base_color);
    smp.pdf = eval_pdf(hit.ffnormal, smp.wi, wo, mat, base_color);

    if smp.pdf > 0.0 {
        smp.weight = bsdf_val * n_dot_l / smp.pdf;
    } else {
        smp.weight = vec3f(0.0);
    }

    return smp;
}

// --- Main Path Tracer ---

// restir_spatial.wgsl (または restir.wgsl)

fn trace_shadow_ray(origin: vec3f, dir: vec3f, dist: f32) -> bool {
    var shadow_rq: ray_query;
    
    // 修正1: TMinを極小にする (1mm -> 0.1mm)
    // 黒点を気にしないなら 0.0 でも良いですが、1e-4 が安全です。
    let t_min = 0.0001; 

    // 修正2: TMaxを「距離 - 固定値」ではなく「距離の99%」にする
    // これにより、距離が1cm未満でも正しく判定されます。
    let t_max = max(dist * 0.999, 0.0);

    if t_min >= t_max {
        // 距離が近すぎる場合は、レイを飛ばさずに「遮蔽なし」とみなすか、
        // 呼び出し元で制御する。ここでは一旦「遮蔽なし」を返す。
        return true;
    }

    rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, t_min, t_max, origin, dir));
    rayQueryProceed(&shadow_rq);
    return rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u;
}

fn reconstruct_geometry_hit(
    custom_data: u32,
    prim_idx: u32,
    bary: vec2f,
    front_face: bool,
    w2o: mat4x3<f32>,
    t: f32,
    ray_origin: vec3f,
    ray_dir: vec3f
) -> HitInfo {
    var hit: HitInfo;

    let raw_id = custom_data;
    let mesh_id = raw_id >> 16u;
    hit.mat_id = raw_id & 0xFFFFu;

    let mesh_info = mesh_infos[mesh_id];
    let idx_offset = mesh_info.index_offset + prim_idx * 3u;

    let i0 = indices[idx_offset + 0u] + mesh_info.vertex_offset;
    let i1 = indices[idx_offset + 1u] + mesh_info.vertex_offset;
    let i2 = indices[idx_offset + 2u] + mesh_info.vertex_offset;

    let v0 = attributes[i0];
    let v1 = attributes[i1];
    let v2 = attributes[i2];

    let n0 = decode_octahedral_normal(v0.normal);
    let n1 = decode_octahedral_normal(v1.normal);
    let n2 = decode_octahedral_normal(v2.normal);

    let u = bary.x;
    let v = bary.y;
    let w = 1.0 - u - v;

    let local_normal = normalize(n0 * w + n1 * u + n2 * v);
    let uv_interp = v0.uv * w + v1.uv * u + v2.uv * v;

    let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);

    hit.normal = normalize(local_normal * m_inv);
    hit.uv = uv_interp;
    hit.front_face = front_face;
    hit.ffnormal = select(-hit.normal, hit.normal, hit.front_face);
    hit.t = t;
    hit.pos = ray_origin + ray_dir * hit.t;

    return hit;
}

fn eval_direct_lighting(hit: HitInfo, wo: vec3f, mat: Material, base_color: vec3f, ls: LightSample, weight: f32) -> vec3f {
    let offset_pos = hit.pos + hit.ffnormal * 0.001;
    let L = normalize(ls.pos - offset_pos);
    let dist = distance(ls.pos, offset_pos);

    let n_dot_l = max(dot(hit.ffnormal, L), 0.0);
    let l_dot_n = max(dot(-L, ls.normal), 0.0);

    if n_dot_l > 0.0 && l_dot_n > 0.0 {
        if trace_shadow_ray(offset_pos, L, dist) {
            let f = eval_bsdf(hit.ffnormal, L, wo, mat, base_color);
            let G = (n_dot_l * l_dot_n) / (dist * dist);
            return ls.emission.rgb * ls.emission.a * f * G * weight;
        }
    }
    return vec3f(0.0);
}

fn trace_path(coord: vec2<i32>, seed: u32) -> PathResult {
    rng_seed = seed;

    var result: PathResult;
    result.radiance = vec3f(0.0);
    result.valid_v1 = false;
    result.v1_pos = vec3f(0.0);
    result.v1_normal = vec3f(0.0);

    let size = textureDimensions(gbuffer_pos);
    let pixel_idx = u32(coord.y) * size.x + u32(coord.x);

    // -------------------------------------------------------------------------
    // 0. Initial State from G-Buffer (Depth = 0)
    // -------------------------------------------------------------------------
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    if pos_w.w < 0.0 {
        return result; // Background
    }
    let normal_w = textureLoad(gbuffer_normal, coord, 0);
    let albedo_raw = textureLoad(gbuffer_albedo, coord, 0);

    var hit: HitInfo;
    hit.pos = pos_w.xyz;
    hit.normal = decode_octahedral_normal(normal_w.xy);
    hit.front_face = true;
    hit.ffnormal = hit.normal;
    // Note: hit.uv, hit.t, hit.mat_id are not fully reconstructed here as we use GBuffer data directly for material

    var mat: Material;
    let mat_id = u32(pos_w.w + 0.1);
    if mat_id < arrayLength(&materials) {
        let mat_static = materials[mat_id];
        mat = mat_static;
        if abs(mat.ior - 1.0) < 0.01 {
            mat.base_color = vec4f(albedo_raw.rgb, 1.0);
        }
    } else {
        mat.base_color = vec4f(albedo_raw.rgb, 1.0);
        mat.roughness = 0.0;
        mat.metallic = albedo_raw.a;
        mat.ior = 1.0;
        mat.light_index = -1;
    }

    var base_color = mat.base_color.rgb;
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    var wo = normalize(camera.view_pos.xyz - hit.pos);

    var next_dir = vec3f(0.0);
    var last_bsdf_pdf = 0.0;
    var previous_was_diffuse = false;

    // -------------------------------------------------------------------------
    // 1. Primary Shading (Emission & ReSTIR NEE)
    // -------------------------------------------------------------------------

    if mat.light_index >= 0 {
        let light = lights[mat.light_index];
        accumulated_color += light.emission.rgb * light.emission.a;
        result.radiance = accumulated_color;
        return result;
    }

    let is_specular = (mat.metallic > 0.01) || (mat.ior > 1.01 || mat.ior < 0.99);
    if !is_specular {
        if camera.num_lights > 0u {
            let light_idx = u32(rand() * f32(camera.num_lights));
            if light_idx < camera.num_lights {
                let ls = sample_light(light_idx);

                let pdf_nee = ls.pdf * (1.0 / f32(camera.num_lights));
                let p_bsdf = eval_pdf(hit.ffnormal, normalize(ls.pos - hit.pos), wo, mat, base_color);
                let mis_weight_nee = pdf_nee / (pdf_nee + p_bsdf);

                let weight = mis_weight_nee / pdf_nee;

                accumulated_color += eval_direct_lighting(hit, wo, mat, base_color, ls, weight) * throughput;
            }
        }
        previous_was_diffuse = true;
    } else {
        previous_was_diffuse = false;
    }

    let sc = sample_bsdf(wo, hit, mat, base_color);
    if sc.weight.x <= 0.0 && sc.weight.y <= 0.0 && sc.weight.z <= 0.0 {
        result.radiance = accumulated_color;
        return result;
    }
    last_bsdf_pdf = sc.pdf;
    throughput *= sc.weight;
    next_dir = sc.wi;


    // -------------------------------------------------------------------------
    // 2. Bounce Loop (Depth 1..MAX_DEPTH)
    // -------------------------------------------------------------------------
    for (var depth = 1u; depth < MAX_DEPTH; depth++) {
        
        // --- Russian Roulette ---
        if depth >= 3u {
            let p = max(throughput.x, max(throughput.y, throughput.z));
            let survival_prob = clamp(p, 0.05, 0.95);
            if rand() > survival_prob { break; }
            throughput /= survival_prob;
        }

        // --- Trace Next Ray ---
        var rq: ray_query;
        let offset_dir = sign(dot(hit.ffnormal, next_dir)) * hit.ffnormal;
        let origin = hit.pos + offset_dir * 0.001;

        rayQueryInitialize(&rq, tlas, RayDesc(0x0u, 0xFFu, 0.001, 100.0, origin, next_dir));
        rayQueryProceed(&rq);
        let committed = rayQueryGetCommittedIntersection(&rq);

        if committed.kind == 0u {
            break;
        }

        // --- Update Hit Info from Geometry using Helper ---
        hit = reconstruct_geometry_hit(
            committed.instance_custom_data,
            committed.primitive_index,
            committed.barycentrics,
            committed.front_face,
            committed.world_to_object,
            committed.t,
            origin,
            next_dir
        );
        // ★Shift Mapping用: 第1バウンスの情報を記録
        if depth == 1u {
            result.valid_v1 = true;
            result.v1_pos = hit.pos;
            result.v1_normal = hit.normal;
        }

        // Update View Dir
        wo = -next_dir;
        
        // Update Material & Base Color
        mat = materials[hit.mat_id];
        let tex_color = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.tex_id), 0.0);
        base_color = mat.base_color.rgb * tex_color.rgb;

        // --- Shading (Secondary) ---

        // 1. Emission (MIS)
        if mat.light_index >= 0 {
            if hit.front_face {
                let light = lights[mat.light_index];
                let Le = light.emission.rgb * light.emission.a;
                var mis_weight = 1.0;
                if previous_was_diffuse {
                    let dist_sq = hit.t * hit.t;
                    let light_cos = max(dot(hit.ffnormal, -wo), 0.0);
                    let p_bsdf = last_bsdf_pdf;
                    let p_nee = (1.0 / light.area) * (dist_sq / light_cos) * (1.0 / f32(camera.num_lights));
                    if light_cos > 0.001 {
                        mis_weight = p_bsdf / (p_bsdf + p_nee);
                    } else { mis_weight = 0.0; }
                }
                accumulated_color += Le * throughput * mis_weight;
            }
            break;
        }

        // 2. NEE (Standard Random Light)
        let is_specular_bounce = (mat.metallic > 0.01) || (mat.ior > 1.01 || mat.ior < 0.99);
        if !is_specular_bounce {
            if camera.num_lights > 0u {
                let light_idx = u32(rand() * f32(camera.num_lights));
                if light_idx < camera.num_lights {
                    let ls = sample_light(light_idx);

                    let pdf_nee = ls.pdf * (1.0 / f32(camera.num_lights));
                    let p_bsdf = eval_pdf(hit.ffnormal, normalize(ls.pos - hit.pos), wo, mat, base_color);
                    let mis_weight_nee = pdf_nee / (pdf_nee + p_bsdf);

                    let weight = mis_weight_nee / pdf_nee;

                    accumulated_color += eval_direct_lighting(hit, wo, mat, base_color, ls, weight) * throughput;
                }
            }
            previous_was_diffuse = true;
        } else {
            previous_was_diffuse = false;
        }

        // 3. BSDF Sample
        let sc_bounce = sample_bsdf(wo, hit, mat, base_color);
        if sc_bounce.weight.x <= 0.0 && sc_bounce.weight.y <= 0.0 && sc_bounce.weight.z <= 0.0 { break; }

        last_bsdf_pdf = sc_bounce.pdf;
        throughput *= sc_bounce.weight;
        next_dir = sc_bounce.wi;
    }

    result.radiance = accumulated_color;
    return result;
}

// --- ReSTIR Helpers ---

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn update_reservoir(r: ptr<function, Reservoir>, seed_cand: u32, w: f32, rnd: f32, cnt: u32, p_hat_new: f32, s_path_new: vec3f) -> bool {
    (*r).w_sum += w;
    (*r).M += cnt;
    if rnd * (*r).w_sum < w {
        (*r).y = seed_cand;
        (*r).p_hat = p_hat_new;
        (*r).s_path = s_path_new;
        return true;
    }
    return false;
}


fn is_valid_neighbor(
    curr_pos: vec3f, curr_normal: vec3f, curr_mat_id: u32,
    prev_pos: vec3f, prev_normal: vec3f, prev_mat_id: u32,
    camera_pos: vec3f
) -> bool {
    // 基本チェック
    if curr_mat_id != prev_mat_id { return false; }

    let mat = materials[curr_mat_id];
    let is_specular = mat.roughness < 0.2 || mat.metallic > 0.8 || (mat.ior > 1.01 || mat.ior < 0.99);

    if is_specular {
        // 鏡面・ガラスの場合は、法線の不一致にめちゃくちゃ厳しくする（例: 3度以内）
        if dot(curr_normal, prev_normal) < 0.998 { return false; }
        
        // 距離の差にも厳しくする
        let dist_diff = distance(curr_pos, prev_pos);
        if dist_diff > 0.01 { return false; }
    } else {
        // 通常の拡散反射素材
        if dot(curr_normal, prev_normal) < 0.995 { return false; } // Stricter normal (approx 5.7 deg)
        
        // Check position distance to avoid leaking between disjoint surfaces
        let dist_to_camera_sq = dot(curr_pos - camera_pos, curr_pos - camera_pos);
        // Stricter depth threshold: 0.1% of distance squared (~3% of distance)
        let threshold = max(0.00001, dist_to_camera_sq * 0.001);
        let dist_diff_sq = dot(curr_pos - prev_pos, curr_pos - prev_pos);
        if dist_diff_sq > threshold { return false; }
    }

    return true;
}

// Jacobianの計算: Reconnection Shift用
// curr_pos: 現在のピクセルの1次ヒット点
// curr_normal: 現在のピクセルの1次ヒット点の法線
// neighbor_v1_pos: 隣接ピクセルのパスにおける「第1バウンス目」のヒット点
// neighbor_pos: 隣接ピクセルの1次ヒット点
// neighbor_normal: 隣接ピクセルの1次ヒット点の法線
fn calculate_jacobian(
    curr_pos: vec3f,
    curr_normal: vec3f,
    curr_albedo: vec3f,
    neighbor_v1_pos: vec3f,
    neighbor_pos: vec3f,
    neighbor_normal: vec3f,
    neighbor_albedo: vec3f
) -> f32 {
    // 共有ヒット点(v1)へのベクトルと距離を計算
    let dir_curr = neighbor_v1_pos - curr_pos;
    let cos_curr = max(dot(curr_normal, normalize(dir_curr)), 0.0);

    let dir_neigh = neighbor_v1_pos - neighbor_pos;
    let cos_neigh = max(dot(neighbor_normal, normalize(dir_neigh)), 0.0);

    if cos_neigh <= 0.001 { return 0.0; } // Stricter cosine check

    // Radiance is invariant along the ray (in vacuum), so we only account for the cosine terms
    // at the integration surfaces (geometry term ratio without distance).
    var jacobian = cos_curr / cos_neigh;

    // Correct for Albedo difference (since p_hat is luminance of radiance ~ albedo)
    let lum_curr = luminance(curr_albedo) + 0.001;
    let lum_neigh = luminance(neighbor_albedo) + 0.001;
    jacobian *= (lum_curr / lum_neigh);

    // ★重要: 境界が光る問題への対策1 (Clamping)
    // 幾何学的に鋭角な場所でJacobianが爆発するのを防ぐ
    jacobian = clamp(jacobian, 0.1, 3.0); // Tightened clamp [0.1, 3.0]

    return jacobian;
}


@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(gbuffer_pos);
    if id.x >= size.x || id.y >= size.y { return; }

    let coord = vec2<i32>(id.xy);
    let pixel_idx = id.y * size.x + id.x;
    
    // RNG Seed
    let seed_init = id.y * size.x + id.x + scene_info.y * 0x12345678u;
    let seed = pcg_hash(seed_init);
    
    // Independent RNG state for spatial loop logic
    var local_seed = seed_init;

    // G-Buffer
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    if pos_w.w < 0.0 {
        // 背景
        var r_out: Reservoir;
        r_out.w_sum = 0.0; r_out.W = 0.0;
        r_out.M = 0u; r_out.y = 0u;
        r_out.p_hat = 0.0;
        r_out.s_path = vec3f(0.0);
        out_reservoirs[pixel_idx] = r_out;
        textureStore(out_tex, coord, vec4f(0.0)); // <--- Fix ghosting
        return;
    }

    let normal_w = textureLoad(gbuffer_normal, coord, 0);
    let normal = decode_octahedral_normal(normal_w.xy);
    let mat_id = u32(pos_w.w + 0.1);
    let albedo = textureLoad(gbuffer_albedo, coord, 0).rgb;

    // 自分のReservoirを読み込む
    var r = in_reservoirs[pixel_idx];
    if r.M > 20u {
        r.w_sum *= 20.0 / f32(r.M);
        r.M = 20u;
    }


    let camera_pos = camera.view_pos.xyz;

    // Spatial Loop (例えば 3~5近傍)
    var num_neighbors = 5u;
    var radius = 10.0; // ピクセル半径 (ノイズの状況に合わせて調整)

    let mat = materials[mat_id];
    if mat.roughness < 0.1 || mat.metallic > 0.9 || mat.ior > 1.01 || mat.ior < 0.99 {
        num_neighbors = 0u;
        // num_neighbors = 2u;
        radius = 2.0; // かなりの近傍のみ探索する
    }

    for (var i = 0u; i < num_neighbors; i++) {
        // ランダムな近傍ピクセルを選ぶ
        let r1 = rand_lcg(&local_seed);
        let r2 = rand_lcg(&local_seed);
        
        // 円盤サンプリング
        let angle = 2.0 * PI * r1;
        let rad = sqrt(r2) * radius;
        let offset = vec2f(cos(angle), sin(angle)) * rad;
        let neighbor_coord = coord + vec2<i32>(offset);

        // 画面外チェック
        if neighbor_coord.x < 0 || neighbor_coord.x >= i32(size.x) || neighbor_coord.y < 0 || neighbor_coord.y >= i32(size.y) {
            continue;
        }

        let neighbor_idx = u32(neighbor_coord.y) * size.x + u32(neighbor_coord.x);
        
        // 近傍のG-Bufferチェック (幾何的類似性)
        let n_pos_w = textureLoad(gbuffer_pos, neighbor_coord, 0);
        if n_pos_w.w < 0.0 { continue; }
        let n_normal_w = textureLoad(gbuffer_normal, neighbor_coord, 0);
        let n_normal = decode_octahedral_normal(n_normal_w.xy);
        let n_mat_id = u32(n_pos_w.w + 0.1);
        let n_albedo = textureLoad(gbuffer_albedo, neighbor_coord, 0).rgb;

        if !is_valid_neighbor(pos_w.xyz, normal, mat_id, n_pos_w.xyz, n_normal, n_mat_id, camera_pos) { continue; }

        var neighbor_r = in_reservoirs[neighbor_idx];

        if neighbor_r.p_hat <= 0.0 { continue; }

        // --- Jacobianの計算 ---
        let jacobian = calculate_jacobian(
            pos_w.xyz,
            normal,
            albedo, // Current albedo
            neighbor_r.s_path, // neighborのパスの第1バウンス点
            n_pos_w.xyz,
            n_normal,
            n_albedo // Neighbor albedo
        );

        // 鏡面素材の場合の追加対策
        let mat = materials[mat_id];
        let is_specular = mat.roughness < 0.1 || mat.metallic > 0.9 || (mat.ior > 1.01 || mat.ior < 0.99);

        if is_specular {
            // 対策: Jacobianが極端に小さい/大きい場合は、接続が物理的に破綻しているとみなして捨てる
            if jacobian < 0.5 || jacobian > 2.0 {
                continue;
            }
        }
        let dir_to_v1 = neighbor_r.s_path - pos_w.xyz;
        let dist_to_v1 = length(dir_to_v1);
        var visible = false;

        if dot(normal, dir_to_v1) > 0.0 {

            if dist_to_v1 > 0.001 {
                let origin = pos_w.xyz;
                let ray_dir = normalize(dir_to_v1);
                let t_max = max(dist_to_v1, 0.0);

                if trace_shadow_ray(origin, ray_dir, t_max) {
                    visible = true;
                }
            } else {
                visible = false;
            }
        }

        if !visible { continue; }

        // jacobian = 1.0;
        // 3. 輝度の補正（ターゲット密度の更新）
        let p_hat_corrected = neighbor_r.p_hat * jacobian;
        let M_new = min(neighbor_r.M, 20u);
        let weight = p_hat_corrected * neighbor_r.W * f32(M_new);

        update_reservoir(&r, neighbor_r.y, weight, rand_lcg(&local_seed), M_new, p_hat_corrected, neighbor_r.s_path);
    }

    // Finalize
    let final_res = trace_path(coord, r.y);
    var final_color = vec3f(0.0);
    let p_hat_final = luminance(final_res.radiance); // 生の輝度
    r.s_path = final_res.v1_pos;

    if p_hat_final > 0.0 {
        // Unbiased Weight Calculation
        var w_unclamped = (1.0 / p_hat_final) * (r.w_sum / f32(r.M));

        r.W = clamp(w_unclamped, 0.0, 3.0);

        final_color = final_res.radiance * r.W;
        r.p_hat = p_hat_final;
    } else {
        r.W = 0.0;
        r.p_hat = 0.0;
    }

    out_reservoirs[pixel_idx] = r;
    textureStore(out_tex, coord, vec4f(final_color, 1.0));
}
