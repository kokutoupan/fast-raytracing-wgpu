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
