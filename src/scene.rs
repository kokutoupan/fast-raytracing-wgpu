use crate::geometry;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

// 頂点データを再エクスポート (state.rs 等での変更を最小限にするため)
pub use crate::geometry::Vertex;

// GPUに送るマテリアルデータ (48バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub emission: [f32; 4],
    pub extra: [f32; 4], // x: type (0=Lambert, 1=Metal, 2=Dielectric), y: fuzz, z: ior, w: padding
}

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
    // Global Resources
    pub global_vertex_buffer: wgpu::Buffer,
    pub global_index_buffer: wgpu::Buffer,
    pub mesh_info_buffer: wgpu::Buffer,

    // Individual BLAS (Needed for TLAS build, but vertices are now global)
    pub plane_blas: wgpu::Blas,
    pub cube_blas: wgpu::Blas,
    pub sphere_blas: wgpu::Blas,

    pub material_buffer: wgpu::Buffer,
}

pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    // 1. 各ジオメトリの生成とBLAS構築
    let plane = geometry::create_plane_blas(device);
    let cube = geometry::create_cube_blas(device);
    let sphere = geometry::create_sphere_blas(device, 3);

    let mut encoder = device.create_command_encoder(&Default::default());

    // Helper to create temporary buffer for BLAS build
    let create_temp_buf = |contents: &[u8]| -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents,
            usage: wgpu::BufferUsages::BLAS_INPUT,
        })
    };

    let plane_v_buf = create_temp_buf(bytemuck::cast_slice(&plane.vertices));
    let plane_i_buf = create_temp_buf(bytemuck::cast_slice(&plane.indices));
    let cube_v_buf = create_temp_buf(bytemuck::cast_slice(&cube.vertices));
    let cube_i_buf = create_temp_buf(bytemuck::cast_slice(&cube.indices));
    let sphere_v_buf = create_temp_buf(bytemuck::cast_slice(&sphere.vertices));
    let sphere_i_buf = create_temp_buf(bytemuck::cast_slice(&sphere.indices));

    // Plane BLAS Build
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &plane.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &plane.desc,
                vertex_buffer: &plane_v_buf,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
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
            blas: &cube.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &cube.desc,
                vertex_buffer: &cube_v_buf,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                index_buffer: Some(&cube_i_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        None,
    );

    // Sphere BLAS Build
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &sphere.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &sphere.desc,
                vertex_buffer: &sphere_v_buf,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                index_buffer: Some(&sphere_i_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        None,
    );

    // 2. Global Buffers & Mesh Info Creation
    let mut global_vertices = Vec::new();
    let mut global_indices = Vec::new();
    let mut mesh_infos = Vec::new();

    let mut current_v_offset = 0;
    let mut current_i_offset = 0;

    // Mesh 0: Plane
    mesh_infos.push(MeshInfo {
        vertex_offset: current_v_offset,
        index_offset: current_i_offset,
        pad: [0; 2],
    });
    global_vertices.extend_from_slice(&plane.vertices);
    global_indices.extend_from_slice(&plane.indices);
    current_v_offset += plane.vertices.len() as u32;
    current_i_offset += plane.indices.len() as u32;

    // Mesh 1: Cube
    mesh_infos.push(MeshInfo {
        vertex_offset: current_v_offset,
        index_offset: current_i_offset,
        pad: [0; 2],
    });
    global_vertices.extend_from_slice(&cube.vertices);
    global_indices.extend_from_slice(&cube.indices);
    current_v_offset += cube.vertices.len() as u32;
    current_i_offset += cube.indices.len() as u32;

    // Mesh 2: Sphere
    mesh_infos.push(MeshInfo {
        vertex_offset: current_v_offset,
        index_offset: current_i_offset,
        pad: [0; 2],
    });
    global_vertices.extend_from_slice(&sphere.vertices);
    global_indices.extend_from_slice(&sphere.indices);

    let global_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Global Vertex Buffer"),
        contents: bytemuck::cast_slice(&global_vertices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let global_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Global Index Buffer"),
        contents: bytemuck::cast_slice(&global_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mesh_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh Info Buffer"),
        contents: bytemuck::cast_slice(&mesh_infos),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 3. TLAS作成 (Cornell Boxの配置)
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("Cornell Box TLAS"),
        max_instances: 8, // 6 (Walls) + 1 (Metal Box) + 1 (Glass Sphere)
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

    // Helper to encode Mesh ID and Material ID
    // Mesh ID: 0=Plane, 1=Cube, 2=Sphere
    let encode_id = |mesh_id: u32, mat_id: u32| (mesh_id << 16) | mat_id;

    // --- Walls (Plane BLAS, Mesh ID = 0) ---
    // Floor (White, Mat 3)
    tlas[0] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Ceiling (White, Mat 3)
    tlas[1] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Back (White, Mat 3)
    tlas[2] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 3),
    );
    // Left (Red, Mat 1)
    tlas[3] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 1),
    );
    // Right (Green, Mat 2)
    tlas[4] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 2),
    );
    // Light (Mat 0)
    tlas[5] = mk_instance(
        &plane.blas,
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        encode_id(0, 0),
    );

    // --- Objects ---

    // Tall Box (Metal, Mat 6 - Rough Metal)
    tlas[6] = mk_instance(
        &cube.blas,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.4)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
        encode_id(1, 6),
    );

    // Glass Sphere (Mesh ID = 2, Mat 4 - Dielectric)
    // Replace Short Box (0.4, -0.75, 0.3)
    tlas[7] = mk_instance(
        &sphere.blas,
        Mat4::from_translation(Vec3::new(0.4, -0.65, 0.3)) * Mat4::from_scale(Vec3::splat(0.75)),
        encode_id(2, 4),
    );

    encoder.build_acceleration_structures(None, Some(&tlas));
    queue.submit(std::iter::once(encoder.finish()));

    // --- 4. マテリアルバッファの作成 ---
    let materials = [
        // 0: Light
        MaterialUniform {
            color: [0.0, 0.0, 0.0, 1.0],
            emission: [10.0, 10.0, 10.0, 1.0],
            extra: [3.0, 0.0, 0.0, 0.0], // Type=3 (Light)
        },
        // 1: Left Wall (Red)
        MaterialUniform {
            color: [0.65, 0.05, 0.05, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 2: Right Wall (Green)
        MaterialUniform {
            color: [0.12, 0.45, 0.15, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 3: White (Floor/Ceil/Back)
        MaterialUniform {
            color: [0.73, 0.73, 0.73, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [0.0, 0.0, 0.0, 0.0],
        },
        // 4: Dielectric (Glass)
        MaterialUniform {
            color: [1.0, 1.0, 1.0, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            extra: [2.0, 0.0, 1.5, 0.0], // Type=2 (Dielectric), IOR=1.5
        },
        // 5: Metal (Silver)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            extra: [1.0, 0.0, 0.0, 0.0], // Type=1 (Metal), Fuzz=0.0
        },
        // 6: ラフな金属
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [1.0, 0.2, 0.0, 0.0], // Type=1 (Metal), Fuzz=0.2
        },
    ];

    let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material Buffer"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    SceneResources {
        tlas,
        global_vertex_buffer,
        global_index_buffer,
        mesh_info_buffer,
        plane_blas: plane.blas,
        cube_blas: cube.blas,
        sphere_blas: sphere.blas,
        material_buffer,
    }
}
