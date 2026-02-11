// GPUに送るマテリアルデータ (64バイト) - Padding adjusted
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub base_color: [f32; 4], // 16 bytes
    pub light_index: i32,     // 4 bytes
    pub _pad0: [u32; 2],      // 8 bytes
    pub transmission: f32,    // 4 bytes

    // --- PBR Parameters (4 bytes each) ---
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,

    // --- Texture IDs ---
    pub tex_id: u32, // Base Color
    pub normal_tex_id: u32,
    pub occlusion_tex_id: u32,
    pub emissive_tex_id: u32,
    pub _pad1: u32, // Padding to 64 bytes
}

#[allow(dead_code)]
impl Material {
    pub fn new(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            light_index: -1,
            _pad0: [0; 2],
            transmission: 0.0,
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.0,
            tex_id: u32::MAX, // Default to "None"
            normal_tex_id: u32::MAX,
            occlusion_tex_id: u32::MAX,
            emissive_tex_id: u32::MAX,
            _pad1: 0,
        }
    }

    pub fn light_index(mut self, index: i32) -> Self {
        self.light_index = index;
        self
    }

    pub fn metallic(mut self, roughness: f32) -> Self {
        self.metallic = 1.0;
        self.roughness = roughness;
        self
    }

    pub fn roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    pub fn glass(mut self, ior: f32) -> Self {
        self.metallic = 0.0;
        self.roughness = 0.0;
        self.ior = ior;
        self.transmission = 1.0;
        self
    }

    pub fn transmission(mut self, transmission: f32) -> Self {
        self.transmission = transmission;
        self
    }

    pub fn texture(mut self, id: u32) -> Self {
        self.tex_id = id;
        self
    }

    pub fn normal_texture(mut self, id: u32) -> Self {
        self.normal_tex_id = id;
        self
    }

    pub fn occlusion_texture(mut self, id: u32) -> Self {
        self.occlusion_tex_id = id;
        self
    }

    pub fn emissive_texture(mut self, id: u32) -> Self {
        self.emissive_tex_id = id;
        self
    }
}
