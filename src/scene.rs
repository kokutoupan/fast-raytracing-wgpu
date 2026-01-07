use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

// GPUに送るマテリアルデータ (48バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub emission: [f32; 4],
    pub extra: [f32; 4], // x: type (0=Lambert, 1=Metal, 2=Dielectric), y: fuzz, z: ior, w: padding
}

#[allow(dead_code)]
pub struct SceneResources {
    pub tlas: wgpu::Tlas,
    // Plane Resources
    pub plane_blas: wgpu::Blas,
    pub plane_vertex_buffer: wgpu::Buffer,
    pub plane_index_buffer: wgpu::Buffer,
    // Cube Resources
    pub cube_blas: wgpu::Blas,
    pub cube_vertex_buffer: wgpu::Buffer,
    pub cube_index_buffer: wgpu::Buffer,

    pub material_buffer: wgpu::Buffer,
    pub plane_blas_desc: wgpu::BlasTriangleGeometrySizeDescriptor,
    pub cube_blas_desc: wgpu::BlasTriangleGeometrySizeDescriptor,
}

// --- ヘルパー関数: 平面(Quad)のBLASを作成 ---
fn create_plane_blas(
    device: &wgpu::Device,
) -> (
    wgpu::Blas,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::BlasTriangleGeometrySizeDescriptor,
) {
    // 1x1 の平面 (XZ平面, 中心0,0)
    let vertices: [f32; 12] = [
        -0.5, 0.0, 0.5, // 左手前
        0.5, 0.0, 0.5, // 右手前
        -0.5, 0.0, -0.5, // 左奥
        0.5, 0.0, -0.5, // 右奥
    ];
    let indices: [u32; 6] = [0, 1, 2, 2, 1, 3]; // Triangle List

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Quad Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });

    let blas_geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 4,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(6),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("Quad BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size.clone()],
        },
    );

    (blas, vertex_buf, index_buf, blas_geo_size)
}

// --- ヘルパー関数: 立方体(Cube)のBLASを作成 ---
fn create_cube_blas(
    device: &wgpu::Device,
) -> (
    wgpu::Blas,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::BlasTriangleGeometrySizeDescriptor,
) {
    // 1x1x1 の立方体 (中心0,0,0)
    // 頂点データ (8個)
    let vertices: [f32; 24] = [
        // 手前
        -0.5, -0.5, 0.5, // 0: 左下前
        0.5, -0.5, 0.5, // 1: 右下前
        0.5, 0.5, 0.5, // 2: 右上前
        -0.5, 0.5, 0.5, // 3: 左上前
        // 奥
        -0.5, -0.5, -0.5, // 4: 左下奥
        0.5, -0.5, -0.5, // 5: 右下奥
        0.5, 0.5, -0.5, // 6: 右上奥
        -0.5, 0.5, -0.5, // 7: 左上奥
    ];

    // インデックス (12トライアングル)
    // 各面2つ
    #[rustfmt::skip]
    let indices: [u32; 36] = [
        // Front (0,1,2,3)
        0, 1, 2, 0, 2, 3,
        // Back (5,4,7,6)
        5, 4, 7, 5, 7, 6,
        // Top (3,2,6,7)
        3, 2, 6, 3, 6, 7,
        // Bottom (4,5,1,0)
        4, 5, 1, 4, 1, 0,
        // Right (1,5,6,2)
        1, 5, 6, 1, 6, 2,
        // Left (4,0,3,7)
        4, 0, 3, 4, 3, 7,
    ];

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cube Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });

    let blas_geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 8,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(36),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("Cube BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size.clone()],
        },
    );

    (blas, vertex_buf, index_buf, blas_geo_size)
}

pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    // 1. BLAS構築 (Plane & Cube)
    let (plane_blas, plane_v_buf, plane_i_buf, plane_desc) = create_plane_blas(device);
    let (cube_blas, cube_v_buf, cube_i_buf, cube_desc) = create_cube_blas(device);

    let mut encoder = device.create_command_encoder(&Default::default());

    // Plane BLAS Build
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &plane_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &plane_desc,
                vertex_buffer: &plane_v_buf,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&plane_i_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        None,
    );

    // Cube BLAS Build
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &cube_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &cube_desc,
                vertex_buffer: &cube_v_buf,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&cube_i_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        None,
    );

    // 2. TLAS作成 (Cornell Boxの配置)
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("Cornell Box TLAS"),
        max_instances: 8, // 6 (Walls) + 2 (Boxes)
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    let mk_instance = |blas: &wgpu::Blas, transform: Mat4, id: u32| {
        let affine = transform.transpose().to_cols_array();
        Some(wgpu::TlasInstance::new(
            blas,
            affine[..12].try_into().unwrap(),
            id,
            0xff,
        ))
    };

    // Helper to encode Geometry Type and Material ID
    // GeoType: 0 = Plane, 1 = Cube
    let encode_id = |geo_type: u32, mat_id: u32| (geo_type << 16) | mat_id;

    // --- Walls (Plane BLAS, GeoType = 0) ---
    // Floor (White, Mat 3)
    tlas[0] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Ceiling (White, Mat 3)
    tlas[1] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Back (White, Mat 3)
    tlas[2] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Left (Red, Mat 1)
    tlas[3] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 1),
    );
    // Right (Green, Mat 2)
    tlas[4] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 2),
    );
    // Light (Mat 0)
    tlas[5] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        encode_id(0, 0),
    );

    // --- Boxes (Cube BLAS, GeoType = 1) ---
    // Tall Box (Metal, Mat 4) -> Right side (Moved from Left)
    tlas[6] = mk_instance(
        &cube_blas,
        Mat4::from_translation(Vec3::new(-0.35, -0.4, -0.3))
            * Mat4::from_rotation_y(0.3)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
        encode_id(1, 4),
    );

    // Short Box (Dielectric, Mat 5) -> Left side (Moved from Right)
    tlas[7] = mk_instance(
        &cube_blas,
        Mat4::from_translation(Vec3::new(0.4, -0.7, 0.3))
            * Mat4::from_rotation_y(-0.3)
            * Mat4::from_scale(Vec3::new(0.6, 0.6, 0.6)),
        encode_id(1, 5),
    );

    encoder.build_acceleration_structures(None, Some(&tlas));
    queue.submit(std::iter::once(encoder.finish()));

    // --- 3. マテリアルバッファの作成 ---
    let materials = [
        // 0: Light
        MaterialUniform {
            color: [0.0, 0.0, 0.0, 1.0],
            emission: [15.0, 15.0, 15.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 1: Left Wall (Red)
        MaterialUniform {
            color: [0.8, 0.1, 0.1, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 2: Right Wall (Green)
        MaterialUniform {
            color: [0.1, 0.8, 0.1, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 3: White (Floor/Ceil/Back)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 5: Dielectric (Glass)
        MaterialUniform {
            color: [1.0, 1.0, 1.0, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            extra: [2.0, 0.0, 1.5, 0.0], // Type=2 (Dielectric), IOR=1.5
        },
        // 4: Metal (Silver)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            extra: [1.0, 0.0, 0.0, 0.0], // Type=1 (Metal), Fuzz=0.0
        },
    ];

    let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material Buffer"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    SceneResources {
        tlas,
        plane_blas,
        plane_vertex_buffer: plane_v_buf,
        plane_index_buffer: plane_i_buf,
        cube_blas,
        cube_vertex_buffer: cube_v_buf,
        cube_index_buffer: cube_i_buf,
        material_buffer,
        plane_blas_desc: plane_desc,
        cube_blas_desc: cube_desc,
    }
}
