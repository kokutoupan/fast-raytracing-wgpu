// GPUに送るマテリアルデータ (64バイト) - Optimized
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    // 0-16 bytes
    pub base_color: [f32; 4],

    // 16-32 bytes
    pub emissive_factor: [f32; 3], // 12 bytes
    pub roughness: f32,            // 4 bytes

    // 32-48 bytes
    pub metallic: f32,
    pub transmission: f32,
    pub ior: f32,
    pub light_index: i32,

    // 48-64 bytes
    // Texture IDs packed as [u16; 2] inside u32
    // tex_info_0: [tex_id (low), normal_tex_id (high)]
    pub tex_info_0: u32,
    // tex_info_1: [occlusion_tex_id (low), emissive_tex_id (high)]
    pub tex_info_1: u32,
    // tex_info_2: [metallic_roughness_tex_id (low), padding (high)]
    pub tex_info_2: u32,

    pub _pad_final: u32, // Ensure full 64 bytes (16-byte align)
}

#[allow(dead_code)]
impl Material {
    pub fn new(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            emissive_factor: [0.0, 0.0, 0.0],
            roughness: 0.5,
            metallic: 0.0,
            transmission: 0.0,
            ior: 1.0,
            light_index: -1,
            // Pack u16::MAX (0xFFFF) for "None"
            tex_info_0: 0xFFFFFFFF, // tex_id, normal
            tex_info_1: 0xFFFFFFFF, // occlusion, emissive
            tex_info_2: 0xFFFFFFFF, // metallic_roughness, padding
            _pad_final: 0,
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

    // Helper to pack u16
    fn pack(current: u32, val: u32, is_high: bool) -> u32 {
        let val_u16 = (val & 0xFFFF) as u32;
        if is_high {
            (current & 0x0000FFFF) | (val_u16 << 16)
        } else {
            (current & 0xFFFF0000) | val_u16
        }
    }

    pub fn texture(mut self, id: u32) -> Self {
        self.tex_info_0 = Self::pack(self.tex_info_0, id, false);
        self
    }

    pub fn normal_texture(mut self, id: u32) -> Self {
        self.tex_info_0 = Self::pack(self.tex_info_0, id, true);
        self
    }

    pub fn occlusion_texture(mut self, id: u32) -> Self {
        self.tex_info_1 = Self::pack(self.tex_info_1, id, false);
        self
    }

    pub fn emissive_texture(mut self, id: u32) -> Self {
        self.tex_info_1 = Self::pack(self.tex_info_1, id, true);
        self
    }

    pub fn metallic_roughness_texture(mut self, id: u32) -> Self {
        self.tex_info_2 = Self::pack(self.tex_info_2, id, false);
        self
    }

    // --- Getters for unpacking values ---
    fn unpack(packed: u32, is_high: bool) -> u32 {
        if is_high {
            (packed >> 16) & 0xFFFF
        } else {
            packed & 0xFFFF
        }
    }

    pub fn get_texture(&self) -> u32 {
        Self::unpack(self.tex_info_0, false)
    }

    pub fn get_normal_texture(&self) -> u32 {
        Self::unpack(self.tex_info_0, true)
    }

    pub fn get_occlusion_texture(&self) -> u32 {
        Self::unpack(self.tex_info_1, false)
    }

    pub fn get_emissive_texture(&self) -> u32 {
        Self::unpack(self.tex_info_1, true)
    }

    pub fn get_metallic_roughness_texture(&self) -> u32 {
        Self::unpack(self.tex_info_2, false)
    }

    pub fn tex_id(&self) -> u32 {
        self.get_texture()
    }
    pub fn normal_tex_id(&self) -> u32 {
        self.get_normal_texture()
    }
    pub fn occlusion_tex_id(&self) -> u32 {
        self.get_occlusion_texture()
    }
    pub fn emissive_tex_id(&self) -> u32 {
        self.get_emissive_texture()
    }
    pub fn metallic_roughness_tex_id(&self) -> u32 {
        self.get_metallic_roughness_texture()
    }

    pub fn emissive_factor(mut self, factor: [f32; 3]) -> Self {
        self.emissive_factor = factor;
        self
    }
}
