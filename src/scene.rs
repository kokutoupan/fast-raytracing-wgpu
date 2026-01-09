use glam::{Mat4, Vec3};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// GPUに送るマテリアルデータ (48バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub emission: [f32; 4],
    pub extra: [f32; 4], // x: type (0=Lambert, 1=Metal, 2=Dielectric), y: fuzz, z: ior, w: padding
}

// 頂点データ (32バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 4],    // vec4f alignment
    pub normal: [f32; 4], // vec4f alignment
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

// --- ヘルパー関数: 平面(Quad)のBLASを作成 ---
fn create_plane_blas(
    device: &wgpu::Device,
) -> (
    wgpu::Blas,
    Vec<Vertex>,
    Vec<u32>,
    wgpu::BlasTriangleGeometrySizeDescriptor,
) {
    // 1x1 の平面 (XZ平面, 中心0,0)
    let vertices = vec![
        Vertex {
            pos: [-0.5, 0.0, 0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 左手前
        Vertex {
            pos: [0.5, 0.0, 0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 右手前
        Vertex {
            pos: [-0.5, 0.0, -0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 左奥
        Vertex {
            pos: [0.5, 0.0, -0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 右奥
    ];
    let indices: Vec<u32> = vec![0, 1, 2, 2, 1, 3]; // Triangle List

    let blas_geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3, // BLAS build still uses Float32x3 (stride 32 handles alignment)
        vertex_count: vertices.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
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

    (blas, vertices, indices, blas_geo_size)
}

// --- ヘルパー関数: 立方体(Cube)のBLASを作成 ---
fn create_cube_blas(
    device: &wgpu::Device,
) -> (
    wgpu::Blas,
    Vec<Vertex>,
    Vec<u32>,
    wgpu::BlasTriangleGeometrySizeDescriptor,
) {
    // 1x1x1 の立方体 (中心0,0,0)
    // 頂点データ (24個: 6面 * 4頂点) - フラットシェーディングのため法線ごとに頂点を分ける
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut v_idx = 0;

    let sides = [
        (
            [0.0, 0.0, 1.0],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ), // Front
        (
            [0.0, 0.0, -1.0],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ), // Back
        (
            [0.0, 1.0, 0.0],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ), // Top
        (
            [0.0, -1.0, 0.0],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
        ), // Bottom
        (
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ), // Right
        (
            [-1.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
        ), // Left
    ];

    for (normal, v0, v1, v2, v3) in sides {
        let n = [normal[0], normal[1], normal[2], 0.0];
        vertices.push(Vertex {
            pos: [v0[0], v0[1], v0[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v1[0], v1[1], v1[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v2[0], v2[1], v2[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v3[0], v3[1], v3[2], 1.0],
            normal: n,
        });

        indices.push(v_idx);
        indices.push(v_idx + 1);
        indices.push(v_idx + 2);
        indices.push(v_idx);
        indices.push(v_idx + 2);
        indices.push(v_idx + 3);

        v_idx += 4;
    }

    let blas_geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertices.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
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

    (blas, vertices, indices, blas_geo_size)
}

// --- ヘルパー関数: 球体(Sphere)のBLASを作成 (UV Sphere) ---
fn create_sphere_blas(
    device: &wgpu::Device,
    subdivisions: u32,
) -> (
    wgpu::Blas,
    Vec<Vertex>,
    Vec<u32>,
    wgpu::BlasTriangleGeometrySizeDescriptor,
) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // 1. 正二十面体の基本定数 (黄金比 phi)
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    // 初期頂点 (半径を 0.5 に調整)
    let mut add_vertex = |p: [f32; 3]| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let n = [p[0] / length, p[1] / length, p[2] / length];
        let pos = [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0];
        vertices.push(Vertex {
            pos,
            normal: [n[0], n[1], n[2], 0.0],
        });
        vertices.len() as u32 - 1
    };

    // 初期12頂点
    add_vertex([-1.0, t, 0.0]);
    add_vertex([1.0, t, 0.0]);
    add_vertex([-1.0, -t, 0.0]);
    add_vertex([1.0, -t, 0.0]);
    add_vertex([0.0, -1.0, t]);
    add_vertex([0.0, 1.0, t]);
    add_vertex([0.0, -1.0, -t]);
    add_vertex([0.0, 1.0, -t]);
    add_vertex([t, 0.0, -1.0]);
    add_vertex([t, 0.0, 1.0]);
    add_vertex([-t, 0.0, -1.0]);
    add_vertex([-t, 0.0, 1.0]);

    // 初期20面 (反時計回り CCW)
    let mut faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // 2. 分割処理
    let mut midpoint_cache = HashMap::new();
    for _ in 0..subdivisions {
        let mut new_faces = Vec::new();
        for tri in faces {
            let v1 = tri[0];
            let v2 = tri[1];
            let v3 = tri[2];

            let a = get_midpoint(v1, v2, &mut vertices, &mut midpoint_cache);
            let b = get_midpoint(v2, v3, &mut vertices, &mut midpoint_cache);
            let c = get_midpoint(v3, v1, &mut vertices, &mut midpoint_cache);

            new_faces.push([v1, a, c]);
            new_faces.push([v2, b, a]);
            new_faces.push([v3, c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }

    for tri in faces {
        indices.extend_from_slice(&tri);
    }

    // 3. wgpu BLAS 構築
    let blas_geo_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3, // Vertex構造体のposに合わせる
        vertex_count: vertices.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("Ico Sphere BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size.clone()],
        },
    );

    (blas, vertices, indices, blas_geo_size)
}

// 中点取得ヘルパー（重複頂点を防ぐ）
fn get_midpoint(
    p1: u32,
    p2: u32,
    vertices: &mut Vec<Vertex>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    if let Some(&index) = cache.get(&key) {
        return index;
    }

    let v1 = vertices[p1 as usize].pos;
    let v2 = vertices[p2 as usize].pos;

    let mid = [
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    ];

    let length = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    let n = [mid[0] / length, mid[1] / length, mid[2] / length];

    vertices.push(Vertex {
        pos: [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0],
        normal: [n[0], n[1], n[2], 0.0],
    });

    let index = vertices.len() as u32 - 1;
    cache.insert(key, index);
    index
}

pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    // 1. 各ジオメトリの生成とBLAS構築
    let (plane_blas, plane_verts, plane_indices, plane_desc) = create_plane_blas(device);
    let (cube_blas, cube_verts, cube_indices, cube_desc) = create_cube_blas(device);
    let (sphere_blas, sphere_verts, sphere_indices, sphere_desc) = create_sphere_blas(device, 3);

    let mut encoder = device.create_command_encoder(&Default::default());

    // Helper to create temporary buffer for BLAS build
    let create_temp_buf = |contents: &[u8]| -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents,
            usage: wgpu::BufferUsages::BLAS_INPUT,
        })
    };

    let plane_v_buf = create_temp_buf(bytemuck::cast_slice(&plane_verts));
    let plane_i_buf = create_temp_buf(bytemuck::cast_slice(&plane_indices));
    let cube_v_buf = create_temp_buf(bytemuck::cast_slice(&cube_verts));
    let cube_i_buf = create_temp_buf(bytemuck::cast_slice(&cube_indices));
    let sphere_v_buf = create_temp_buf(bytemuck::cast_slice(&sphere_verts));
    let sphere_i_buf = create_temp_buf(bytemuck::cast_slice(&sphere_indices));

    // Plane BLAS Build
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &plane_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &plane_desc,
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
            blas: &cube_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &cube_desc,
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
            blas: &sphere_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &sphere_desc,
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
    global_vertices.extend_from_slice(&plane_verts);
    global_indices.extend_from_slice(&plane_indices);
    current_v_offset += plane_verts.len() as u32;
    current_i_offset += plane_indices.len() as u32;

    // Mesh 1: Cube
    mesh_infos.push(MeshInfo {
        vertex_offset: current_v_offset,
        index_offset: current_i_offset,
        pad: [0; 2],
    });
    global_vertices.extend_from_slice(&cube_verts);
    global_indices.extend_from_slice(&cube_indices);
    current_v_offset += cube_verts.len() as u32;
    current_i_offset += cube_indices.len() as u32;

    // Mesh 2: Sphere
    mesh_infos.push(MeshInfo {
        vertex_offset: current_v_offset,
        index_offset: current_i_offset,
        pad: [0; 2],
    });
    global_vertices.extend_from_slice(&sphere_verts);
    global_indices.extend_from_slice(&sphere_indices);

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
        max_instances: 10, // 6 (Walls) + 2 (Boxes) + 1 (Sphere) + 1 (Lens)
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
    // Mesh ID: 0=Plane, 1=Cube, 23Sphere
    let encode_id = |mesh_id: u32, mat_id: u32| (mesh_id << 16) | mat_id;

    // --- Walls (Plane BLAS, Mesh ID = 0) ---
    // Floor (White, Mat 7)
    tlas[0] = mk_instance(
        &plane_blas,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
        encode_id(0, 7),
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

    // --- Boxes (Cube BLAS, Mesh ID = 1) ---
    // 0.002ずらして影を落とさないようにする

    // Tall Box (Dielectric, Mat 4)
    tlas[6] = mk_instance(
        &cube_blas,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.3)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
        encode_id(1, 4),
    );

    // Short Box (Metal, Mat 5)
    tlas[7] = mk_instance(
        &cube_blas,
        Mat4::from_translation(Vec3::new(0.4, -0.7 + 0.002, 0.3))
            * Mat4::from_rotation_y(-0.3)
            * Mat4::from_scale(Vec3::new(0.6, 0.6, 0.6)),
        encode_id(1, 7),
    );

    // Sphere (Mesh ID = 2, Mat 6)
    // Move inside Tall Glass Box (-0.35, -0.4, -0.3)
    // Box width is 0.6, so sphere scale 0.25 (Diameter 0.5) fits safely.
    tlas[8] = mk_instance(
        &sphere_blas,
        Mat4::from_translation(Vec3::new(-0.35, -0.4, -0.3)) * Mat4::from_scale(Vec3::splat(0.25)),
        encode_id(2, 6),
    );

    // Lens (Sphere BLAS, Mesh ID = 2, Mat 8)
    // Position: (0.0, 0.0, 2.0) - in front of camera (Z=3)
    // Scale: (1.5, 1.5, 0.1) - oblate spheroid acting as a convex lens
    tlas[9] = mk_instance(
        &sphere_blas,
        Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0))
            * Mat4::from_scale(Vec3::new(1.5, 1.5, 0.15)),
        encode_id(2, 8),
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
        // 6: Sphere Material (Blue Light)
        MaterialUniform {
            color: [0.0, 0.0, 0.0, 1.0],
            emission: [0.1, 0.1, 10.0, 1.0], // Bright Blue Light
            extra: [3.0, 0.0, 0.0, 0.0],     // Type=3 (Light)
        },
        // 7: ラフな金属
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            extra: [1.0, 0.2, 0.0, 0.0], // Type=1 (Metal), Fuzz=0.2
        },
        // 8: Lens Glass (Dielectric)
        MaterialUniform {
            color: [1.0, 1.0, 1.0, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
            extra: [2.0, 0.0, 1.5, 0.0], // Type=2 (Dielectric), IOR=1.5
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
        plane_blas,
        cube_blas,
        sphere_blas,
        material_buffer,
    }
}
