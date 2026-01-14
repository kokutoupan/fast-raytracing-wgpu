pub mod builder;
pub mod light;
pub mod material;
pub mod resources;

use crate::geometry::{self, Vertex};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub use builder::SceneBuilder;
pub use material::Material;
pub use resources::SceneResources;

pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // 1. 各ジオメトリの生成
    let plane_geo = geometry::create_plane_blas(device);
    let cube_geo = geometry::create_cube_blas(device);
    let sphere_geo = geometry::create_sphere_blas(device, 3);
    let crystal_geo = geometry::create_crystal_blas(device);

    // BLAS Build
    let mut encoder = device.create_command_encoder(&Default::default());

    let build_blas = |encoder: &mut wgpu::CommandEncoder, geo: &geometry::Geometry| {
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&geo.vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&geo.indices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });

        encoder.build_acceleration_structures(
            Some(&wgpu::BlasBuildEntry {
                blas: &geo.blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &geo.desc,
                        vertex_buffer: &v_buf,
                        first_vertex: 0,
                        vertex_stride: std::mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&i_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            None,
        );
    };

    build_blas(&mut encoder, &plane_geo);
    build_blas(&mut encoder, &cube_geo);
    build_blas(&mut encoder, &sphere_geo);
    build_blas(&mut encoder, &crystal_geo);

    queue.submit(std::iter::once(encoder.finish()));

    // 2. メッシュとマテリアルの登録
    let plane_id = builder.add_mesh(plane_geo);
    let cube_id = builder.add_mesh(cube_geo);
    let sphere_id = builder.add_mesh(sphere_geo);
    let crystal_id = builder.add_mesh(crystal_geo);

    // マテリアル定義
    let mat_light = builder.add_material(
        Material::new([0.0, 0.0, 0.0, 1.0])
            .emission([1.0, 1.0, 1.0], 10.0)
            .texture(0),
    );
    let mat_red = builder.add_material(Material::new([0.65, 0.05, 0.05, 1.0]).texture(0));
    let mat_green = builder.add_material(Material::new([0.12, 0.45, 0.15, 1.0]).texture(0));
    let mat_white = builder.add_material(Material::new([0.73, 0.73, 0.73, 1.0]).texture(0));
    let mat_checker = builder.add_material(Material::new([0.73, 0.73, 0.73, 1.0]).texture(1));
    let mat_rough_metal =
        builder.add_material(Material::new([0.8, 0.8, 0.8, 1.0]).metallic(0.2).texture(1));

    // 追加: 球体ライト (強い発光)
    let mat_sphere_light = builder.add_material(
        Material::new([1.0, 1.0, 1.0, 1.0])
            .emission([0.02, 0.02, 0.9], 10.0) // 少し暖色
            .texture(0),
    );

    // 追加: クリスタル (屈折)
    let mat_crystal = builder.add_material(
        Material::new([0.5, 0.8, 1.0, 1.0]) // 薄い水色
            .glass(1.5)
            .texture(0),
    );

    // 3. インスタンスの配置
    // Floor (Checker)
    builder.add_instance(
        plane_id,
        mat_checker,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
    );
    // Ceiling (White)
    builder.add_instance(
        plane_id,
        mat_white,
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.0)),
    );
    // Back (White)
    builder.add_instance(
        plane_id,
        mat_white,
        Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
    );
    // Left (Red)
    builder.add_instance(
        plane_id,
        mat_red,
        Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
    );
    // Right (Green)
    builder.add_instance(
        plane_id,
        mat_green,
        Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
    );
    // Main Light (Ceiling)
    builder.add_instance(
        plane_id,
        mat_light,
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
    );
    // Main Light:Light info
    builder.add_quad_light(
        Vec3::new(0.0, 0.99, 0.0).into(),
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 1.0, 1.0, 10.0],
    );

    // クリスタル (正八面体) - ガラス球のあった場所に移動し拡大
    let crystal_pos = Vec3::new(0.4, -0.5, 0.3);
    builder.add_instance(
        crystal_id,
        mat_crystal,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.5)),
    );

    // クリスタル内部の光
    builder.add_instance(
        sphere_id,
        mat_sphere_light,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.1)),
    );
    // 球体光源の情報
    builder.add_sphere_light(
        Vec3::new(0.4, -0.5, 0.3).into(),
        0.1,
        [0.02, 0.02, 0.9, 10.0],
    );

    // Tall Box (Rough Metal)
    builder.add_instance(
        cube_id,
        mat_rough_metal,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.4)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
    );

    builder.build(device, queue)
}
