enable wgpu_ray_query;

// --- Constants ---
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
    tangent: vec4f,
}

struct PathResult {
    radiance: vec3f,
    valid_v1: bool,
    v1_pos: vec3f,
    v1_normal: vec3f,
}

// --- Bindings ---

// Group 0: G-Buffer & Scene context
@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_motion: texture_2d<f32>;

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
@group(0) @binding(14) var prev_gbuffer_albedo: texture_2d<f32>;

// ... (existing code) ...



// ... (inside main) ...

@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var textures: texture_2d_array<f32>;

// Group 2: Reservoirs
@group(2) @binding(0) var<storage, read_write> prev_reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> curr_reservoirs: array<Reservoir>;

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

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness; // For G1, we use alpha = roughness (not roughness^2 if remapped, but standard GGX usually uses alpha. Here assuming roughness param is linear roughness)
    // However, widely used Smith-Schlick approximation uses k = alpha^2 / 2 ? No, for IBL k=alpha^2/2, for Direct k=(alpha+1)^2/8.
    // Let's implement EXACT G1 for GGX to match VNDF sampling perfectly.

    let a2 = roughness * roughness;
    return 2.0 * n_dot_v / (n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v));
}

fn geometry_smith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_l, roughness) * geometry_schlick_ggx(n_dot_v, roughness);
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
    if mat.transmission > 0.01 { return 0.0; }

    if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return 0.0; }

    // Unified PBR
    let F0 = mix(vec3f(0.04), base_color, mat.metallic);
    let F = fresnel_schlick(F0, max(dot(normal, wo), 0.0));
    let lum_spec = luminance(F);
    let lum_diff = luminance(base_color * (1.0 - mat.metallic));
    let prob_spec = clamp(lum_spec / (lum_spec + lum_diff + 0.0001), 0.001, 0.999);

    // Specular PDF
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(normal, h), 0.0);
    let d = ndf_ggx(n_dot_h, mat.roughness);
    let g1 = geometry_schlick_ggx(n_dot_v, mat.roughness); // Use G1(v)
    let pdf_spec = (d * g1) / (4.0 * n_dot_v);

    // Diffuse PDF
    let pdf_diff = max(n_dot_l, 0.0) / PI;

    return prob_spec * pdf_spec + (1.0 - prob_spec) * pdf_diff;
}

fn eval_bsdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material, base_color: vec3f) -> vec3f {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    // Glass (Delta)
    if mat.transmission > 0.01 { return vec3f(0.0); }

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
    if mat.transmission > 0.01 {
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
    let prob_spec = clamp(lum_spec / (lum_spec + lum_diff + 0.0001), 0.001, 0.999);

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

fn trace_shadow_ray(origin: vec3f, dir: vec3f, dist: f32) -> bool {
    var shadow_rq: ray_query;
    let t_max = max(dist * 0.999, 0.0);
    rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, 0.001, t_max, origin, dir));
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
    
    let t0 = v0.tangent;
    let t1 = v1.tangent;
    let t2 = v2.tangent;

    let u = bary.x;
    let v = bary.y;
    let w = 1.0 - u - v;

    let local_normal = normalize(n0 * w + n1 * u + n2 * v);
    let local_tangent = normalize(t0.xyz * w + t1.xyz * u + t2.xyz * v);
    
    let uv_interp = v0.uv * w + v1.uv * u + v2.uv * v;

    let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);

    hit.normal = normalize(local_normal * m_inv);
    // Transform tangent to world space
    let tangent_w = normalize(local_tangent * m_inv);
    hit.tangent = vec4f(tangent_w, t0.w); // Preserving sign from v0 (assume constant)

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
        mat.base_color = vec4f(albedo_raw.rgb, 1.0);
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

    let is_glass = (mat.transmission > 0.01);
    let is_smooth_metal = (mat.metallic > 0.01) && (mat.roughness < 0.05);
    let is_specular = is_glass || is_smooth_metal;
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
        var tex_color = vec4f(1.0);
        if mat.tex_id != 4294967295u {
            tex_color = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.tex_id), 0.0);
        }

        var occlusion = 1.0;
        if mat.occlusion_tex_id != 4294967295u {
            occlusion = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.occlusion_tex_id), 0.0).r;
        }
        base_color = mat.base_color.rgb * tex_color.rgb * occlusion;

        // --- Normal Mapping (Perturb hit.ffnormal) ---
        if mat.normal_tex_id != 4294967295u {
            let normal_map = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.normal_tex_id), 0.0).rgb;
            let normal_local = normalize(normal_map * 2.0 - 1.0);
            
            let tangent_sign = hit.tangent.w;
            let tangent_w = hit.tangent.xyz;
            let N_ff = hit.ffnormal;

            // Re-orthogonalize T against N_ff
            let T_ff = normalize(tangent_w - N_ff * dot(N_ff, tangent_w));
            let B_ff = normalize(cross(N_ff, T_ff)) * tangent_sign;
            let TBN_ff = mat3x3f(T_ff, B_ff, N_ff);
            
            hit.ffnormal = normalize(TBN_ff * normal_local);
        }

        // --- Emission (via Texture) ---
        // If material has emission texture BUT NOT explicitly an efficient light (light_index == -1), add it here.
        if mat.light_index == -1 && mat.emissive_tex_id != 4294967295u {
            let emissive_col = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.emissive_tex_id), 0.0).rgb;
            accumulated_color += emissive_col * throughput;
        }

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
        let is_glass_bounce = (mat.transmission > 0.01);
        let is_smooth_metal_bounce = (mat.metallic > 0.01) && (mat.roughness < 0.05);
        let is_specular_bounce = is_glass_bounce || is_smooth_metal_bounce;
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
    curr_pos: vec3f, curr_normal: vec3f, curr_mat: u32,
    prev_pos: vec3f, prev_normal: vec3f, prev_mat: u32,
    camera_pos: vec3f,
) -> bool {
    // 1. Material ID Check
    if curr_mat != prev_mat { return false; }
    
    // 2. Normal Check (Stricter)
    if dot(curr_normal, prev_normal) < 0.99 { return false; }

    // 3. Position Check (Relative to camera distance)
    let dist_diff_sq = dot(curr_pos - prev_pos, curr_pos - prev_pos);
    let dist_to_camera_sq = dot(curr_pos - camera_pos, curr_pos - camera_pos);
    
    // Stricter threshold: 0.1% of distance squared
    let threshold = max(0.00001, dist_to_camera_sq * 0.001);

    if dist_diff_sq > threshold { return false; }
    return true;
}

// --- RNG Helper ---
fn rand_lcg(state: ptr<function, u32>) -> f32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(gbuffer_pos);
    if id.x >= size.x || id.y >= size.y { return; }

    let coord = vec2<i32>(id.xy);
    let pixel_idx = id.x + id.y * size.x;
    
    // Seed RNG
    let seed_base = pixel_idx + camera.frame_count * 927163u;
    let seed_candidate = pcg_hash(seed_base);
    
    // Independent RNG for logic
    var local_seed = seed_base;

    // G-Buffer 読み込み
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    if pos_w.w < 0.0 {
        // 背景ならReservoirをクリアして終了
        var r: Reservoir;
        r.w_sum = 0.0; r.W = 0.0; r.M = 0u; r.y = 0u;
        curr_reservoirs[pixel_idx] = r;
        return;
    }

    var r: Reservoir;
    r.w_sum = 0.0;
    r.M = 0u;
    r.W = 0.0;
    r.y = 0u;
    r.p_hat = 0.0;
    r.s_path = vec3f(0.0);

    // =========================================================
    // Phase 1: Initial Candidate Generation (1本生成)
    // =========================================================
    
    // 新しいシードでパスをトレースし、その明るさ(p_hat)を評価
    let path_result = trace_path(coord, seed_candidate);
    let p_hat = luminance(path_result.radiance);

    // Reservoirに登録 (RIS)
    // ここでは候補が1つだけなので、必ず採用される (rnd=0.5)
    // weight w = p_hat / source_pdf (source_pdf = 1.0 とみなす)
    // M accumulation: 1 sample
    update_reservoir(&r, seed_candidate, p_hat, 0.5, 1u, p_hat, path_result.v1_pos);

    // 初期Wの計算
    if p_hat > 0.0 {
        r.W = 1.0; // 1候補だけなのでWは1.0 (w_sum/p_hat/M = p_hat/p_hat/1 = 1)
    } else {
        r.W = 0.0;
    }

    // =========================================================
    // Phase 2: Temporal Reuse (時間的再利用)
    // =========================================================

    let motion = textureLoad(gbuffer_motion, coord, 0);
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
    let prev_uv = uv + motion.xy;
    let prev_id_xy = vec2u(prev_uv * vec2f(size));

    let MAX_RESERVOIR_M_TEMPORAL = 16u;
    
    // 画面内チェック
    if prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0 {
        let prev_pixel_idx = prev_id_xy.y * size.x + prev_id_xy.x;

        // 過去のG-Buffer情報をロードして幾何的一貫性をチェック
        let prev_pos_data = textureLoad(prev_gbuffer_pos, prev_id_xy, 0);
        let prev_normal_data = textureLoad(prev_gbuffer_normal, prev_id_xy, 0);
        let prev_normal = decode_octahedral_normal(prev_normal_data.xy);
        let prev_mat_id = u32(prev_pos_data.w + 0.1);

        let curr_normal_data = textureLoad(gbuffer_normal, coord, 0);
        let curr_normal = decode_octahedral_normal(curr_normal_data.xy);
        let curr_mat_id = u32(pos_w.w + 0.1);
        let mat = materials[curr_mat_id];
        // FIX: Flashing Floor/Highlights
        // Disable temporal reuse for specular/glossy surfaces because highlights are view-dependent
        // and reusing p_hat from previous frame (different view angle) is invalid.
        let is_specular = mat.roughness < 0.2 || mat.metallic > 0.8 || (mat.transmission > 0.01);

        if is_valid_neighbor(
            pos_w.xyz, curr_normal, curr_mat_id,
            prev_pos_data.xyz, prev_normal, prev_mat_id,
            camera.view_pos.xyz
        ) && !is_specular {
            var prev_r = prev_reservoirs[prev_pixel_idx];
            
            // ALBEDO CORRECTION (No Replay)
            let curr_albedo = textureLoad(gbuffer_albedo, coord, 0).rgb;
            let prev_albedo = textureLoad(prev_gbuffer_albedo, prev_id_xy, 0).rgb;
            let l_curr = luminance(curr_albedo) + 0.001;
            let l_prev = luminance(prev_albedo) + 0.001;
            let albedo_ratio = l_curr / l_prev;

            // FIX: Exploding weights
            // If albedo changed significantly (disocclusion or texture boundary), reject history.
            if albedo_ratio < 3.0 && albedo_ratio > 0.33 {
                // Use stored p_hat corrected by albedo change
                let p_hat_new = prev_r.p_hat * albedo_ratio;

                if p_hat_new > 0.0 {
                    let clamped_M = min(prev_r.M, MAX_RESERVOIR_M_TEMPORAL);
                    let w_prev = p_hat_new * prev_r.W * f32(clamped_M);

                    update_reservoir(&r, prev_r.y, w_prev, rand_lcg(&local_seed), clamped_M, p_hat_new, prev_r.s_path);
                }
            }
        }
    }

    // =========================================================
    // Phase 3: Finalize (Wの更新)
    // =========================================================
    
    // Finalize W using cached p_hat
    let p_hat_final = r.p_hat;

    if p_hat_final > 0.0 {
        r.W = (1.0 / p_hat_final) * (r.w_sum / f32(r.M));
        // r.p_hat is already set
    } else {
        r.W = 0.0;
        r.p_hat = 0.0;
    }

    curr_reservoirs[pixel_idx] = r;
}
