// GPUに送るマテリアルデータ (48バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub base_color: [f32; 4], // 16 bytes (RGB + A/Padding)
    pub emission: [f32; 4],   // 16 bytes (RGB + Strength)

    // --- PBR Parameters (4 bytes each) ---
    pub roughness: f32, // 0.0 ~ 1.0
    pub metallic: f32,  // 0.0 or 1.0 (or blend)
    pub ior: f32,       // Index of Refraction (1.45 etc)
    pub tex_id: u32,    // Texture Array Index
}

impl Material {
    pub fn new(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            emission: [0.0; 4],
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.0,
            tex_id: 0, // Default White
        }
    }

    pub fn emission(mut self, emission: [f32; 3], strength: f32) -> Self {
        self.emission = [emission[0], emission[1], emission[2], strength];
        self
    }

    pub fn metallic(mut self, roughness: f32) -> Self {
        self.metallic = 1.0;
        self.roughness = roughness;
        self
    }

    pub fn glass(mut self, ior: f32) -> Self {
        self.metallic = 0.0;
        self.roughness = 0.0;
        self.ior = ior;
        self
    }

    pub fn texture(mut self, id: u32) -> Self {
        self.tex_id = id;
        self
    }
}
