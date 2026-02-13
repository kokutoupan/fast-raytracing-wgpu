// メッシュ情報 (16バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInfo {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub pad: [u32; 2],
}

#[allow(dead_code)]
pub struct SceneResources {
    pub tlas: wgpu::Tlas,
    pub global_attribute_buffer: wgpu::Buffer,
    pub global_index_buffer: wgpu::Buffer,
    pub mesh_info_buffer: wgpu::Buffer,
    pub blases: Vec<wgpu::Blas>,
    pub material_buffer: wgpu::Buffer,
    pub light_buffer: wgpu::Buffer,
    pub num_lights: u32,
    pub color_texture_view: wgpu::TextureView,
    pub data_texture_view: wgpu::TextureView,
}
