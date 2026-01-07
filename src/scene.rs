use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

// GPUに送るマテリアルデータ (32バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub emission: [f32; 4],
}

pub struct SceneResources {
    pub tlas: wgpu::Tlas,
    pub blas: wgpu::Blas,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub blas_geo_size: wgpu::BlasTriangleGeometrySizeDescriptor,
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

pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    // 1. BLAS構築 (平面)
    let (blas, v_buf, i_buf, blas_desc) = create_plane_blas(device);

    // BLASのビルドコマンド発行
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        Some(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas_desc,
                vertex_buffer: &v_buf,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&i_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        None, // TLAS構築は後でやるのでここではNone (依存関係がないなら同時でもいいが、TLASでBLAS使うので)
    );
    // BLAS構築完了を待つ必要は本来あるが、同じキューでTLAS構築するなら順序保証されるはず
    // ただし wgpu の仕様上、TLAS構築時に BLAS が valid である必要がある。
    // build_acceleration_structures でまとめてやるのが一般的だが、
    // ここでは単純化のため一度 submit してしまう手もある。
    // 今回は TLAS構築も同じ encoder でやってみる。 wgpuは依存関係を追跡してくれるお利口さんなはず。

    // 2. TLAS作成 (Cornell Boxの配置)
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("Cornell Box TLAS"),
        max_instances: 6, // 床, 天井, 奥, 左, 右, ライト
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    // インスタンス作成用クロージャ
    // id: 0=Light, 1=Left(Red), 2=Right(Green), 3=White(Floor/Ceil/Back)
    let mk_instance = |transform: Mat4, id: u32| {
        let affine = transform.transpose().to_cols_array();
        Some(wgpu::TlasInstance::new(
            &blas,
            affine[..12].try_into().unwrap(),
            id,
            0xff,
        ))
    };

    // 部屋のサイズは 2.0 ( -1.0 ~ 1.0 ) と仮定
    // 床 (白)
    tlas[0] = mk_instance(
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
        3,
    );
    // 天井 (白)
    tlas[1] = mk_instance(
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.0)),
        3,
    );
    // 奥壁 (白)
    tlas[2] = mk_instance(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        3,
    );
    // 左壁 (赤)
    tlas[3] = mk_instance(
        Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        1,
    );
    // 右壁 (緑)
    tlas[4] = mk_instance(
        Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        2,
    );
    // ライト (発光) - 天井の少し下
    tlas[5] = mk_instance(
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        0,
    );

    encoder.build_acceleration_structures(
        None, // BLASは上で構築リクエスト済み
        Some(&tlas),
    );

    queue.submit(std::iter::once(encoder.finish()));

    // --- 3. マテリアルバッファの作成 ---
    // インデックス順: 0:床, 1:天井, 2:奥, 3:左, 4:右, 5:ライト
    let materials = [
        // 0: 床 (白)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
        },
        // 1: 天井 (白)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
        },
        // 2: 奥壁 (白)
        MaterialUniform {
            color: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
        },
        // 3: 左壁 (赤)
        MaterialUniform {
            color: [0.8, 0.1, 0.1, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
        },
        // 4: 右壁 (緑)
        MaterialUniform {
            color: [0.1, 0.8, 0.1, 1.0],
            emission: [0.0, 0.0, 0.0, 0.0],
        },
        // 5: ライト (発光) - 色は白、Emissionを強力に
        MaterialUniform {
            color: [0.0, 0.0, 0.0, 1.0],
            emission: [15.0, 15.0, 15.0, 1.0],
        },
    ];

    let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material Buffer"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    SceneResources {
        tlas,
        blas,
        vertex_buffer: v_buf,
        index_buffer: i_buf,
        material_buffer,
        blas_geo_size: blas_desc,
    }
}
