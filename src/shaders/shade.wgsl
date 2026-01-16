enable wgpu_ray_query;

// --- Constants ---
const PI = 3.14159265359;
const MAX_DEPTH = 8u; // User requested "Max Depth" loop

// --- Structs ---
struct Camera {
    view_proj: mat4x4f,
    view_inv: mat4x4f,
    proj_inv: mat4x4f,
    view_pos: vec4f,
    prev_view_proj: mat4x4f,
    frame_count: u32,       // Added to match raytrace.wgsl if needed, or use uniform hack
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

// --- Bindings ---
@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var out_color: texture_storage_2d<rgba32float, write>;

@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<storage, read> reservoirs: array<Reservoir>;
@group(0) @binding(7) var tlas: acceleration_structure;
@group(0) @binding(8) var<storage, read> materials: array<Material>;
@group(0) @binding(9) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(10) var<storage, read> indices: array<u32>;
@group(0) @binding(11) var<storage, read> mesh_infos: array<MeshInfo>;

// --- RNG ---
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

fn random_unit_vector() -> vec3f {
    let z = rand() * 2.0 - 1.0;
    let a = rand() * 2.0 * PI;
    let r = sqrt(1.0 - z * z);
    let x = r * cos(a);
    let y = r * sin(a);
    return vec3f(x, y, z);
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

// Group 1: Textures
@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var textures: texture_2d_array<f32>;

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
fn eval_pdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material) -> f32 {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    if mat.metallic > 0.01 {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return 0.0; }
        let h = normalize(wi + wo);
        let n_dot_h = max(dot(normal, h), 0.0);
        let d = ndf_ggx(n_dot_h, mat.roughness);
        let k = (mat.roughness * mat.roughness) / 2.0;
        let g1 = geometry_schlick_ggx(n_dot_v, k);
        return (d * g1) / (4.0 * n_dot_v);
    }
    if mat.ior > 1.01 || mat.ior < 0.99 { return 0.0; } // Delta
    return max(n_dot_l, 0.0) / PI;
}

fn eval_bsdf(normal: vec3f, wi: vec3f, wo: vec3f, mat: Material, base_color: vec3f) -> vec3f {
    let n_dot_l = dot(normal, wi);
    let n_dot_v = dot(normal, wo);

    if mat.metallic > 0.01 {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 { return vec3f(0.0); }
        let h = normalize(wi + wo);
        let n_dot_h = max(dot(normal, h), 0.0);
        let h_dot_v = max(dot(h, wo), 0.0);
        let D = ndf_ggx(n_dot_h, mat.roughness);
        let G = geometry_smith(n_dot_l, n_dot_v, mat.roughness);
        let F = fresnel_schlick(base_color, h_dot_v);
        let numerator = D * G * F;
        let denominator = 4.0 * n_dot_l * n_dot_v;
        return numerator / max(denominator, 0.001);
    }
    if mat.ior > 1.01 || mat.ior < 0.99 { return vec3f(0.0); }
    return base_color / PI;
}

fn sample_bsdf(wo: vec3f, hit: HitInfo, mat: Material, base_color: vec3f) -> BsdfSample {
    var smp: BsdfSample;
    smp.is_delta = false;

    if mat.metallic > 0.01 {
        let tbn = make_orthonormal_basis(hit.ffnormal);
        let wo_local = transpose(tbn) * wo;
        let r_uv = vec2f(rand(), rand());
        let wm_local = sample_ggx_vndf(wo_local, mat.roughness, r_uv);
        let wm = tbn * wm_local;
        smp.wi = reflect(-wo, wm);

        let n_dot_l = dot(hit.ffnormal, smp.wi);
        let n_dot_v = dot(hit.ffnormal, wo);

        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            smp.weight = vec3f(0.0);
            smp.pdf = 0.0;
            return smp;
        }

        smp.pdf = eval_pdf(hit.ffnormal, smp.wi, wo, mat);
        let F = fresnel_schlick(base_color, dot(wo, wm));
        let k = (mat.roughness * mat.roughness) / 2.0;
        let G1_l = geometry_schlick_ggx(n_dot_l, k);
        smp.weight = F * G1_l;
        return smp;
    }
    
    // Glass
    if mat.ior > 1.01 || mat.ior < 0.99 {
        smp.is_delta = true;
        smp.pdf = 0.0;
        let refraction_ratio = select(mat.ior, 1.0 / mat.ior, hit.front_face);
        let cos_theta = min(dot(wo, hit.ffnormal), 1.0);
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > rand() { // Simplified Fresnel prob
            smp.wi = reflect(-wo, hit.ffnormal);
        } else {
            smp.wi = refract(-wo, hit.ffnormal, refraction_ratio);
        }
        smp.weight = base_color;
        return smp;
    }

    // Lambert
    smp.wi = normalize(hit.ffnormal + random_unit_vector());
    let n_dot_l = max(dot(hit.ffnormal, smp.wi), 0.0);
    smp.pdf = n_dot_l / PI;
    if smp.pdf > 0.0 { smp.weight = base_color; } else { smp.weight = vec3f(0.0); }
    return smp;
}

// --- Main Path Tracer ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_color);
    if id.x >= size.x || id.y >= size.y { return; }

    let coord = vec2<i32>(id.xy);
    let pixel_idx = id.y * size.x + id.x;
    
    // Seed RNG
    init_rng(id.xy, size.x, camera.frame_count); // Ensure camera has frame_count

    // Initial Path State from G-Buffer
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    if pos_w.w < 0.0 {
        textureStore(out_color, coord, vec4f(0.0, 0.0, 0.0, 1.0)); // Background
        return;
    }
    let normal_w = textureLoad(gbuffer_normal, coord, 0);
    let albedo_raw = textureLoad(gbuffer_albedo, coord, 0);

    var hit: HitInfo;
    hit.pos = pos_w.xyz;
    hit.normal = normal_w.xyz;
    hit.front_face = true; // Assumed for primary
    hit.ffnormal = hit.normal;
    // We don't have primary Hit UV or MatID in standard G-Buffer properly separate?
    // albedo_raw.a is metallic/mat_id? In gbuffer.wgsl it was metallic.
    // normal_w.w is roughness.
    // ShadePass assumes we just use these values directly for primary shading.
    // For Material consistency in loop, we'll reconstruct a temporary "Material" for depth=0
    
    // Loop Variables
    var accumulated_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    var wo = normalize(camera.view_pos.xyz - hit.pos); // View dir goes TO scene? No, wo is usually TO camera.

    var previous_was_diffuse = false;
    var last_bsdf_pdf = 0.0;
    
    // Construct Primary Material Proxy
    var mat_primary: Material;
    let mat_id = u32(pos_w.w + 0.1);
    if mat_id < arrayLength(&materials) {
        let mat_static = materials[mat_id];
        mat_primary = mat_static;
        // Override with G-Buffer sampled values (which include texture modulation)
        mat_primary.base_color = vec4f(albedo_raw.rgb, 1.0);
        mat_primary.roughness = normal_w.w;
        mat_primary.metallic = albedo_raw.a;
    } else {
        // Fallback
        mat_primary.base_color = vec4f(albedo_raw.rgb, 1.0);
        mat_primary.roughness = normal_w.w;
        mat_primary.metallic = albedo_raw.a;
        mat_primary.ior = 1.0;
        mat_primary.light_index = -1;
    }
    
    // --- Loop ---
    for (var depth = 0u; depth < MAX_DEPTH; depth++) {
        // Resolve Material
        var mat = mat_primary;
        
        // Use the hit info's mat_id for secondary bounces
        if depth > 0u {
            mat = materials[hit.mat_id];
        }

        var base_color = mat.base_color.rgb;

        // Texture Sampling
        // Primary (depth=0): Already handled via GBuffer override.
        // Secondary (depth>0): Manual sampling using interpolated UVs.
        if depth > 0u {
            let tex_color = textureSampleLevel(textures, tex_sampler, hit.uv, i32(mat.tex_id), 0.0);
            base_color *= tex_color.rgb;
        }

        // 1. Emission (MIS)
        if mat.light_index >= 0 {
            if !hit.front_face {
               break;
            }
            let light = lights[mat.light_index];
            let Le = light.emission.rgb * light.emission.a;
            var mis_weight = 1.0;
            if previous_was_diffuse {
                let dist_sq = hit.t * hit.t;
                let light_cos = max(dot(hit.ffnormal, -wo), 0.0); // -wo is ray dir coming in? No WO is ToCamera. 
                 // We need incoming direction `wi` from previous bounce.
                 // In loop, `wo` tracks "view direction" for current hit.
                 // Ray direction was -wo.
                let p_bsdf = last_bsdf_pdf;
                let p_nee = (1.0 / light.area) * (dist_sq / light_cos) * (1.0 / f32(camera.num_lights));
                if light_cos > 0.001 {
                    mis_weight = p_bsdf / (p_bsdf + p_nee);
                } else { mis_weight = 0.0; }
            }
            accumulated_color += Le * throughput * mis_weight;
            break; // Stop at light
        }
        
        // 2. NEE
        let is_specular = (mat.metallic > 0.01) || (mat.ior > 1.01 || mat.ior < 0.99);
        if !is_specular {
            var Ld = vec3f(0.0);
            if depth == 0u {
                 // ** ReSTIR NEE **
                let r = reservoirs[pixel_idx];
                if r.W > 0.0 {
                    let light_idx = r.y;
                    let light = lights[light_idx];
                     // Sample Light Surface
                     // Note: We use pixel index for RNG to be consistent? Or rand()? 
                     // Using rand() is fine since we init RNG at start.
                    let ls = sample_light(light_idx); // Uses global RNG

                    let offset_pos = hit.pos + hit.ffnormal * 0.001;
                    let L = normalize(ls.pos - offset_pos);
                    let dist = distance(ls.pos, offset_pos);

                    let n_dot_l = max(dot(hit.ffnormal, L), 0.0);
                    let l_dot_n = max(dot(-L, ls.normal), 0.0);

                    if n_dot_l > 0.0 && l_dot_n > 0.0 {
                        var shadow_rq: ray_query;
                        rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0x1u, 0.001, dist - 0.01, offset_pos, L));
                        rayQueryProceed(&shadow_rq);
                        if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
                            let f = eval_bsdf(hit.ffnormal, L, wo, mat, base_color);
                            let G = (n_dot_l * l_dot_n) / (dist * dist);
                             // ReSTIR Weight `r.W` includes 1/pdf_choice.
                             // We multiply by Area (1/pdf_sample) because we sampled surface?
                             // ReSTIR assumes point light usually. If we sample area, we multiply by Area.
                             // `ls.pdf` is 1/Area.
                            Ld = ls.emission.rgb * ls.emission.a * f * G * r.W * (1.0 / ls.pdf);
                        }
                    }
                }
                previous_was_diffuse = true;
            } else {
                 // ** Standard NEE ** (Random Light)
                if camera.num_lights > 0u {
                    let light_idx = u32(rand() * f32(camera.num_lights));
                    if light_idx < camera.num_lights {
                        let ls = sample_light(light_idx);
                        let offset_pos = hit.pos + hit.ffnormal * 0.001;
                        let L = normalize(ls.pos - offset_pos);
                        let dist = distance(ls.pos, offset_pos);
                        let n_dot_l = max(dot(hit.ffnormal, L), 0.0);
                        let l_dot_n = max(dot(-L, ls.normal), 0.0);

                        if n_dot_l > 0.0 && l_dot_n > 0.0 {
                            var shadow_rq: ray_query;
                            rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0xFFu, 0.001, dist - 0.01, offset_pos, L));
                            rayQueryProceed(&shadow_rq);
                            if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
                                let f = eval_bsdf(hit.ffnormal, L, wo, mat, base_color);
                                let G = (n_dot_l * l_dot_n) / (dist * dist);
                                let pdf_nee = ls.pdf * (1.0 / f32(camera.num_lights)); 
                                 // MIS Weight for NEE
                                 // p_bsdf? We haven't sampled BSDF yet.
                                 // Standard NEE estimator: Le * f * G / pdf_nee.
                                 // With MIS: weight = p_nee / (p_nee + p_bsdf).
                                 // But commonly we just use NEE + BSDF(weighted).
                                 // Let's use simple NEE here with heuristic?
                                 // If we do NEE, we weight it against potential BSDF hit.
                                let p_bsdf = eval_pdf(hit.ffnormal, L, wo, mat);
                                let mis_weight = pdf_nee / (pdf_nee + p_bsdf);

                                Ld = ls.emission.rgb * ls.emission.a * f * G * mis_weight / pdf_nee;
                            }
                        }
                    }
                }
                previous_was_diffuse = true;
            }
            accumulated_color += Ld * throughput;
        } else {
            previous_was_diffuse = false;
        }

        // 3. BSDF Sample & Bounce
        let sc = sample_bsdf(wo, hit, mat, base_color);
        if sc.weight.x <= 0.0 && sc.weight.y <= 0.0 && sc.weight.z <= 0.0 { break; }

        last_bsdf_pdf = sc.pdf;
        throughput *= sc.weight;
        let next_dir = sc.wi;
        
        // Trace Next Ray
        if depth < MAX_DEPTH - 1u {
            var rq: ray_query;
            // Offset logic: Push away from surface in direction of travel
            // Reflection (dot>0) -> Push Out (+ffnormal)
            // Refraction (dot<0) -> Push In (-ffnormal)
            let offset_dir = sign(dot(hit.ffnormal, next_dir)) * hit.ffnormal;
            let origin = hit.pos + offset_dir * 0.001;

            rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, next_dir));
            rayQueryProceed(&rq);
            let committed = rayQueryGetCommittedIntersection(&rq);

            if committed.kind == 0u {
                 // Miss (Sky color?)
                 break;
            }
             
             // Update Hit Info
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

            let u = committed.barycentrics.x;
            let v = committed.barycentrics.y;
            let w = 1.0 - u - v;

            let local_normal = normalize(v0.normal.xyz * w + v1.normal.xyz * u + v2.normal.xyz * v);
            let uv_interp = v0.uv.xy * w + v1.uv.xy * u + v2.uv.xy * v;

            let w2o = committed.world_to_object;
            let m_inv = mat3x3f(w2o[0], w2o[1], w2o[2]);
            hit.normal = normalize(local_normal * m_inv);
            hit.uv = uv_interp;
            hit.front_face = committed.front_face;
            hit.ffnormal = select(-hit.normal, hit.normal, hit.front_face);
            hit.t = committed.t;
            hit.pos = origin + next_dir * hit.t; // Use the actual ray origin!

            wo = -next_dir;
        }
    }

    textureStore(out_color, coord, vec4f(accumulated_color, 1.0));
}
