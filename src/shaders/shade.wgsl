enable wgpu_ray_query;

// --- Structs ---
struct Camera {
    view_proj: mat4x4f,
    view_inv: mat4x4f,
    proj_inv: mat4x4f,
    view_pos: vec4f,
    prev_view_proj: mat4x4f,
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

struct SurfaceParams {
    base_color: vec4f,
    roughness: f32,
    metallic: f32,
    emission: vec4f,
}

// Matches Rust MaterialUniform (64 bytes)
struct Material {
    base_color: vec4f,
    emission: vec4f,
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
    light_index: i32,
    texture_id: i32,
    pad1: u32,
    pad2: u32,
}

struct BsdfSample {
    wi: vec3f,
    pdf: f32,
    weight: vec3f,
}

// --- Bindings ---
@group(0) @binding(0) var gbuffer_pos: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var out_color: texture_storage_2d<rgba32float, write>;

// New Bindings for ReSTIR Shading
@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<storage, read> reservoirs: array<Reservoir>;
@group(0) @binding(7) var tlas: acceleration_structure;
@group(0) @binding(8) var<storage, read> materials: array<Material>;

// --- Constants ---
const PI = 3.14159265359;

// --- RNG (PCG Hash) ---
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

// --- Helper Functions ---
fn fresnel_schlick(f0: vec3f, v_dot_h: f32) -> vec3f {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

fn ndf_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.141592 * d * d);
}

fn geometry_schlick_ggx(n_dot_v: f32, k: f32) -> f32 {
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let ggx1 = geometry_schlick_ggx(n_dot_l, k);
    let ggx2 = geometry_schlick_ggx(n_dot_v, k);
    return ggx1 * ggx2;
}

fn make_orthonormal_basis(n: vec3f) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3f(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3f(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(tangent, bitangent, n);
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

    let Ne = normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

fn sample_bsdf(wo: vec3f, normal: vec3f, mat: SurfaceParams, base_color: vec3f) -> BsdfSample {
    var smp: BsdfSample;
    
    // Metal (GGX) or Glossy
    if mat.metallic > 0.01 || mat.roughness < 1.0 {
        let tbn = make_orthonormal_basis(normal);
        let wo_local = transpose(tbn) * wo;

        let r_uv = vec2f(rand(), rand());
        let wm_local = sample_ggx_vndf(wo_local, mat.roughness, r_uv);
        let wm = tbn * wm_local;

        smp.wi = reflect(-wo, wm);

        let n_dot_l = dot(normal, smp.wi);
        let n_dot_v = dot(normal, wo);

        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            smp.weight = vec3f(0.0);
            smp.pdf = 0.0;
            return smp;
        }
        
        // P_wi = D * G1 / (4 * n.v)
        let k = (mat.roughness * mat.roughness) / 2.0;
        let d = ndf_ggx(max(dot(normal, wm), 0.0), mat.roughness);
        let g1 = geometry_schlick_ggx(n_dot_v, k);
        smp.pdf = (d * g1) / (4.0 * n_dot_v);

        let h = normalize(smp.wi + wo);
        let F = fresnel_schlick(base_color, max(dot(wo, h), 0.0));
        let G = geometry_smith(n_dot_l, n_dot_v, mat.roughness);
        // Note: D is calculated again for weight logic if needed, but D terms cancel out in weight?
        // Weight = F * (G / G1)
        let g1_l = geometry_schlick_ggx(n_dot_l, k);
        smp.weight = F * g1_l;

        return smp;
    }

    // Lambert
    let r1 = rand();
    let r2 = rand();
    let phi = 2.0 * PI * r1;
    let theta = acos(sqrt(r2));
    let x = sin(theta) * cos(phi);
    let y = sin(theta) * sin(phi);
    let z = cos(theta);

    let tbn = make_orthonormal_basis(normal);
    smp.wi = tbn * vec3f(x, y, z);
    smp.pdf = z / PI; // cos(theta) / PI
    smp.weight = base_color;

    return smp;
}

fn sample_quad(light: Light, u: vec2f) -> vec3f {
    return light.position + (2.0 * u.x - 1.0) * light.u + (2.0 * u.y - 1.0) * light.v;
}

fn sample_sphere(light: Light, u: vec2f) -> vec3f {
    let z = 1.0 - 2.0 * u.x;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * u.y;
    let x = r * cos(phi);
    let y = r * sin(phi);
    return light.position + vec3f(x, y, z) * light.v.x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_color);
    if id.x >= size.x || id.y >= size.y { return; }

    let coord = vec2<i32>(id.xy);
    let pixel_idx = id.y * size.x + id.x;

    // Read G-Buffer
    let pos_w = textureLoad(gbuffer_pos, coord, 0);
    let normal_w = textureLoad(gbuffer_normal, coord, 0);
    let albedo_raw = textureLoad(gbuffer_albedo, coord, 0);

    if pos_w.w < 0.0 {
        textureStore(out_color, coord, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let pos = pos_w.xyz;
    let normal = normal_w.xyz;
    let albedo = albedo_raw.rgb;
    let metallic = albedo_raw.a;
    let roughness = normal_w.w;

    // View Direction
    let wo = normalize(pos - camera.view_pos.xyz);

    // Read Reservoir
    let r = reservoirs[pixel_idx];
    
    // Accumulate Light
    var color = vec3f(0.0);
    
    // Check if we have a valid sample
    if r.W > 0.0 {
        let light_idx = r.y;
        let light = lights[light_idx];

        // Sample Point on Light Surface provided ReSTIR selected this light
        // Use RNG seeded by pixel + frame (or just pixel for now if ReSTIR is noisy enough)
        // We use a different seed than BSDF to decorrelate? 
        // Re-init RNG is cheap.
        // Wait, camera.view_pos.w is 1.0 generally. 
        // Use id.x + id.y * size.x as seed + a constant offset.
        // We already have a global RNG function that uses `rng_seed`.
        // We need to init it.
        // Let's just use the same init as BSDF but draw earlier.

        init_rng(id.xy, size.x, u32(r.W * 1000.0) + id.x); // Temporary seed strategy

        let r_sample = vec2f(rand(), rand());
        var light_pos_sample = vec3f(0.0);
        var light_normal = vec3f(0.0, 1.0, 0.0);

        if light.type_ == 0u { // Quad
            light_pos_sample = sample_quad(light, r_sample);
            light_normal = normalize(cross(light.u, light.v));
        } else { // Sphere
            light_pos_sample = sample_sphere(light, r_sample);
            light_normal = normalize(light_pos_sample - light.position);
        }

        let offset_pos = pos + normal * 0.01;
        let L_vec = light_pos_sample - offset_pos;
        let dist_sq = dot(L_vec, L_vec);
        let dist = sqrt(dist_sq);
        let L = L_vec / dist;
        let NdotL = max(dot(normal, L), 0.0);
        
        // Visibility Check (Shadow Ray)
        var shadow_rq: ray_query;
        let t_max = max(0.0, dist - 0.01);
        // Using RayQuery to check visibility
        rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0x1u, 0.01, t_max, offset_pos, L));
        rayQueryProceed(&shadow_rq);
        
        // If unoccluded
        if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
             // Evaluate Light Intensity
             // For area lights, we just use Emission. Attenuation is 1/dist^2 solid angle term.
             // If we sample surface uniformly by Area, PDF = 1/Area.
             // Estimator = Le * G * V / (1/Area) = Le * G * V * Area.
             // G = cos(theta_l) * cos(theta_shading) / dist^2.
             
             // Check if we hit the front face of the light? (for Quad)
            let l_dot_n_light = dot(-L, light_normal);

            if l_dot_n_light > 0.0 || light.type_ == 1u { // Sphere emits formatted
                let G_light = select(l_dot_n_light, 1.0, light.type_ == 1u); // Sphere always emits towards point? Or strictly cosine? 
                // Usually sphere area light emission is uniform in all directions?
                // Let's assume cosine at surface for physical consistency?
                // Actually, standard sphere lights emit uniformly... let's stick to simple model.
                // ReSTIR formulation in `target_pdf` didn't check light normal.

                let attenuation = 1.0 / max(dist_sq, 0.01);
                let radiance = light.emission.rgb * light.emission.a * attenuation;
                
                // Evaluate BRDF
                let H = normalize(wo + L);
                let NdotH = max(dot(normal, H), 0.0);
                let VdotH = max(dot(wo, H), 0.0);
                let NdotV = max(dot(normal, wo), 0.0);

                let F0 = mix(vec3f(0.04), albedo, metallic);
                let F = fresnel_schlick(F0, VdotH);
                let D = ndf_ggx(NdotH, roughness);
                let G = geometry_smith(NdotL, NdotV, roughness);

                let numerator = D * G * F;
                let denominator = 4.0 * NdotL * NdotV;
                let specular = numerator / max(denominator, 0.001);

                let kS = F;
                let kD = (vec3f(1.0) - kS) * (1.0 - metallic);

                let diffuse = kD * albedo / 3.141592;
                let brdf = diffuse + specular; 
                
                // Final Color Contribution
                // Note: r.W essentially contains 1/pdf_ris.
                // If we assume RIS picked the 'light' proportional to something,
                // and now we integrate over the light area.
                // We multiply by Area because we sampled 1 point from Area pdf (1/Area).
                // And G_light factor (cosine at light) if Quad.

                let light_cosine = select(max(l_dot_n_light, 0.0), 1.0, light.type_ == 1u);

                color += brdf * radiance * NdotL * r.W * light.area * light_cosine;
             }
        }
    }

    // --- BSDF Sampling (Add specularity/reflection) ---
    // Make SurfaceParams struct from G-Buffer data for helper
    var mat_proxy: SurfaceParams;
    mat_proxy.base_color = vec4f(albedo, 1.0);
    mat_proxy.roughness = roughness;
    mat_proxy.metallic = metallic;
    mat_proxy.emission = vec4f(0.0);
    
    // Init RNG
    init_rng(id.xy, size.x, u32(r.W * 1000.0) + id.x);

    let bsdf_sample = sample_bsdf(wo, normal, mat_proxy, albedo);

    if bsdf_sample.pdf > 0.0 {
        var ray_bsdf: ray_query;
        // Trace against everything (Mask 0xFF) to find Lights
        rayQueryInitialize(&ray_bsdf, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, pos + normal * 0.001, bsdf_sample.wi));
        rayQueryProceed(&ray_bsdf);

        let committed = rayQueryGetCommittedIntersection(&ray_bsdf);
        if committed.kind == 0u {
             // Miss
        } else {
             // Hit - Is it a light?
            let raw_id = committed.instance_custom_data;
            let mat_id = raw_id & 0xFFFFu;
            let hit_mat = materials[mat_id];

            if hit_mat.light_index >= 0 {
                let light = lights[hit_mat.light_index];
                 
                 // Calculate lighting contribution
                 // Evaluate Light PDF/Emission analytically
                 // Need Light Normal.
                var light_normal = vec3f(0.0, 1.0, 0.0);

                if light.type_ == 0u { // Quad
                    light_normal = normalize(cross(light.u, light.v));
                 } else { // Sphere
                    let hit_t = committed.t;
                    let hit_pos = pos + bsdf_sample.wi * hit_t;
                    light_normal = normalize(hit_pos - light.position);
                 }

                let l_dot_n_light = max(dot(-bsdf_sample.wi, light_normal), 0.0);

                if l_dot_n_light > 0.0 {
                    // BSDF Weight = f * cos / pdf
                    let weight = bsdf_sample.weight;
                    let Le = light.emission.rgb * light.emission.a;
                    
                    // Simple Additive MIS (Combined Estimator: NEE + BSDF)
                    // We assume ReSTIR is approx NEE.
                    // color += Le * weight;
                    // BUT if we want true MIS, we need balance heuristic.
                    // For now, naive addition as requested.
                    color += Le * weight;
                }
            }
        }
    }

    textureStore(out_color, coord, vec4f(color, 1.0));
}
