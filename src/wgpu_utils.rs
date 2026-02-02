use wgpu::util::DeviceExt;

pub fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub fn create_buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn get_padded_bytes_per_row(width: u32) -> u32 {
    let unpadded_bytes_per_row = width * 4;
    let align = 256;
    let padding = (align - unpadded_bytes_per_row % align) % align;
    unpadded_bytes_per_row + padding
}

#[allow(dead_code)]
pub fn generate_white_texture_data(dim: u32) -> Vec<u8> {
    vec![255; (dim * dim * 4) as usize]
}

#[allow(dead_code)]
pub fn generate_checkerboard_texture_data(dim: u32, tiles: u32) -> Vec<u8> {
    let mut data = vec![0u8; (dim * dim * 4) as usize];
    let tile_size = dim / tiles;
    for y in 0..dim {
        for x in 0..dim {
            let is_white = ((x / tile_size) + (y / tile_size)) % 2 == 0;
            let color = if is_white { 255 } else { 0 };
            let idx = ((y * dim + x) * 4) as usize;
            data[idx] = color;
            data[idx + 1] = color;
            data[idx + 2] = color;
            data[idx + 3] = 255;
        }
    }
    data
}
