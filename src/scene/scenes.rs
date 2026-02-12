use super::builder::SceneBuilder;
use super::loader;
use super::material::Material;
use super::resources::SceneResources;
use crate::geometry;
use glam::{Mat4, Vec3};

#[allow(dead_code)]
pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // 1. 各ジオメトリの生成
    let plane_geo = geometry::create_plane_blas(device);
    let cube_geo = geometry::create_cube_blas(device);
    let sphere_geo = geometry::create_sphere_blas(device, 3);
    let crystal_geo = geometry::create_crystal_blas(device);

    // BLAS Build (まとめて実行)
    SceneBuilder::build_blases(
        device,
        queue,
        &[&plane_geo, &cube_geo, &sphere_geo, &crystal_geo],
    );

    // 2. メッシュとマテリアルの登録
    let plane_id = builder.add_mesh(plane_geo);
    let cube_id = builder.add_mesh(cube_geo);
    let sphere_id = builder.add_mesh(sphere_geo);
    let crystal_id = builder.add_mesh(crystal_geo);

    // マテリアル定義
    let mat_red = builder.add_material(Material::new([0.65, 0.05, 0.05, 1.0]).texture(0));
    let mat_green = builder.add_material(Material::new([0.12, 0.45, 0.15, 1.0]).texture(0));
    let mat_white = builder.add_material(Material::new([0.73, 0.73, 0.73, 1.0]).texture(0));
    let mat_checker = builder.add_material(
        Material::new([0.73, 0.73, 0.73, 1.0])
            .roughness(0.99) // Matte
            .texture(1),
    );
    let mat_rough_metal = builder.add_material(
        Material::new([0.8, 0.8, 0.8, 1.0])
            .metallic(0.01)
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
        0x1,
    );
    // Ceiling (White)
    builder.add_instance(
        plane_id,
        mat_white,
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.0)),
        0x1,
    );
    // Back (White)
    builder.add_instance(
        plane_id,
        mat_white,
        Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        0x1,
    );
    // Left (Red)
    builder.add_instance(
        plane_id,
        mat_red,
        Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        0x1,
    );
    // Right (Green)
    builder.add_instance(
        plane_id,
        mat_green,
        Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.0)),
        0x1,
    );
    // Main Light (Ceiling)
    builder.register_quad_light(
        plane_id,
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        [1.0, 1.0, 1.0], // Color
        10.0,
    );

    // クリスタル (正八面体) - ガラス球のあった場所に移動し拡大
    let crystal_pos = Vec3::new(0.4, -0.5, 0.3);
    builder.add_instance(
        crystal_id,
        mat_crystal,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.5)),
        0x1,
    );

    // クリスタル内部の光 (Sphere Light)
    builder.register_sphere_light(
        sphere_id,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.1)),
        [0.02, 0.02, 0.9],
        10.0,
    );

    // Tall Box (Rough Metal)
    builder.add_instance(
        cube_id,
        mat_rough_metal,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.4)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
        0x1,
    );

    builder.build(device, queue)
}

#[allow(dead_code)]
pub fn create_restir_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // 1. 各ジオメトリの生成
    let plane_geo = geometry::create_plane_blas(device);
    let sphere_geo = geometry::create_sphere_blas(device, 2); // 軽めの分割で
    let cube_geo = geometry::create_cube_blas(device);

    // BLAS Build
    SceneBuilder::build_blases(device, queue, &[&plane_geo, &sphere_geo, &cube_geo]);

    // 2. メッシュの登録
    let plane_id = builder.add_mesh(plane_geo);
    let sphere_id = builder.add_mesh(sphere_geo);
    let cube_id = builder.add_mesh(cube_geo);

    // 3. マテリアル
    let mat_floor = builder.add_material(
        Material::new([0.73, 0.73, 0.73, 1.0])
            .roughness(0.99)
            .texture(0),
    );
    let mat_wall = builder.add_material(
        Material::new([0.73, 0.73, 0.73, 1.0])
            .roughness(0.99)
            .texture(0),
    );
    let mat_metal =
        builder.add_material(Material::new([1.0, 1.0, 1.0, 1.0]).metallic(0.2).texture(0));

    // 4. 床と壁
    // Floor
    builder.add_instance(
        plane_id,
        mat_floor,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(10.0)),
        0x1,
    );
    // Back wall
    builder.add_instance(
        plane_id,
        mat_wall,
        Mat4::from_translation(Vec3::new(0.0, 5.0, -5.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(10.0)),
        0x1,
    );

    // 5. 多数の光源 (Grid状に配置)
    let rows = 10;
    let cols = 10;
    let spacing = 1.0;
    let light_radius = 0.05;
    let emission_strength = 20.0;

    for r in 0..rows {
        for c in 0..cols {
            let x = (c as f32 - cols as f32 / 2.0) * spacing;
            let z = (r as f32 - rows as f32 / 2.0) * spacing;
            let y = -0.9; // 少し床から浮かす

            // 各ライトに別の色を割り当てる (虹色っぽく)
            let hue = (r * cols + c) as f32 / (rows * cols) as f32;
            let color = hsv_to_rgb(hue, 0.8, 1.0);
            let emission = [color[0], color[1], color[2], emission_strength];

            let mat_id = builder.add_material(
                Material::new([color[0], color[1], color[2], 1.0])
                    .light_index((r * cols + c) as i32)
                    .emissive_factor([
                        color[0] * emission_strength,
                        color[1] * emission_strength,
                        color[2] * emission_strength,
                    ])
                    .texture(0),
            );

            // インスタンス(可視)
            builder.add_instance(
                sphere_id,
                mat_id,
                Mat4::from_translation(Vec3::new(x, y, z))
                    * Mat4::from_scale(Vec3::splat(light_radius)),
                0x2,
            );

            // 光源データ
            builder.add_sphere_light([x, y, z], light_radius, emission);
        }
    }

    // 6. 受光部としてのオブジェクト
    builder.add_instance(
        cube_id,
        mat_metal,
        Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)) * Mat4::from_scale(Vec3::splat(0.5)),
        0x1,
    );

    builder.build(device, queue)
}

#[allow(dead_code)]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [r + m, g + m, b + m]
}

#[allow(dead_code)]
pub fn create_gltf_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    file_path: &str,
    model_transform: Mat4,
    light_transform: Mat4,
) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // 1. 各ジオメトリの生成
    let plane_geo = geometry::create_plane_blas(device);
    // ライト用のジオメトリ (Quad)
    let light_geo = geometry::create_plane_blas(device);

    // GLTF読み込み
    match loader::load_gltf(file_path, device) {
        Ok((geos, mats, images, gltf_mat_indices)) => {
            println!(
                "Loaded {}: {} geometries, {} materials, {} images",
                file_path,
                geos.len(),
                mats.len(),
                images.len()
            );
            // 2. BLAS Build
            let mut all_geos = vec![&plane_geo, &light_geo];
            for geo in &geos {
                all_geos.push(geo);
            }
            SceneBuilder::build_blases(device, queue, &all_geos);

            // 3. Add default meshes/materials and instances
            // Plane
            let plane_id = builder.add_mesh(plane_geo);
            // Light
            let light_mesh_id = builder.add_mesh(light_geo);

            // Materials
            let mat_floor = builder.add_material(
                Material::new([0.73, 0.73, 0.73, 1.0])
                    .roughness(0.99)
                    .texture(0)
                    .normal_texture(2) // Flat
                    .occlusion_texture(0) // White
                    .emissive_texture(3), // Black
            );
            // Floor Instance
            builder.add_instance(
                plane_id,
                mat_floor,
                Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0))
                    * Mat4::from_scale(Vec3::splat(10.0)),
                0x1,
            );

            // Light Instance and Data
            builder.register_quad_light(light_mesh_id, light_transform, [1.0, 1.0, 1.0], 15.0);

            // 4. Add GLTF Resources
            let gltf_mat_ids = builder.add_gltf_materials(mats, images);
            let gltf_mesh_ids = builder.add_gltf_meshes(geos);

            // 5. Add GLTF Instances
            builder.add_gltf_instances(
                &gltf_mesh_ids,
                &gltf_mat_ids,
                &gltf_mat_indices,
                model_transform,
            );
        }
        Err(e) => {
            eprintln!("Failed to load {}: {:?}", file_path, e);
        }
    }

    builder.build(device, queue)
}

#[allow(dead_code)]
pub fn create_avocado_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    create_gltf_scene(
        device,
        queue,
        "assets/models/Avocado.glb",
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(20.0)),
        Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(1.0)),
    )
}

#[allow(dead_code)]
pub fn create_damaged_helmet_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    create_gltf_scene(
        device,
        queue,
        "assets/models/DamagedHelmet.glb",
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI / 2.0)
            * Mat4::from_scale(Vec3::splat(1.0)),
        Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(1.0)),
    )
}

#[allow(dead_code)]
pub fn create_multi_material_model_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> SceneResources {
    create_gltf_scene(
        device,
        queue,
        "assets/models/AliciaSolid.vrm",
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))
            * Mat4::from_scale(Vec3::splat(0.5))
            * Mat4::from_rotation_y(std::f32::consts::PI),
        Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(1.0)),
    )
}
