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

struct Material {
    base_color: vec4f,
    roughness: f32,
    metallic: f32,
    emission: vec4f,
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

        let L_vec = light.position - pos;
        let dist_sq = dot(L_vec, L_vec);
        let dist = sqrt(dist_sq);
        let L = L_vec / dist;
        let NdotL = max(dot(normal, L), 0.0);

        // Visibility Check (Shadow Ray)
        var shadow_rq: ray_query;
        let t_max = max(0.0, dist - 0.01);
        // Using RayQuery to check visibility
        rayQueryInitialize(&shadow_rq, tlas, RayDesc(0x4u, 0x1u, 0.01, t_max, pos + normal * 0.001, L));
        rayQueryProceed(&shadow_rq);
        
        // If unoccluded
        if rayQueryGetCommittedIntersection(&shadow_rq).kind == 0u {
             // Evaluate Light Intensity
            let attenuation = 1.0 / max(dist_sq, 0.01);
            let radiance = light.emission.rgb * light.emission.a * attenuation;
             
             // Evaluate BRDF
             // Simple PBR (Cook-Torrance)
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
             
             // Final Color Contribution = BRDF * Radiance * NdotL * ReservoirWeight * Area
            color += brdf * radiance * NdotL * r.W * light.area;
        }
    }

    textureStore(out_color, coord, vec4f(color, 1.0));
}
