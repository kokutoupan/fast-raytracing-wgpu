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
    let mat_light = builder.add_material(
        Material::new([0.0, 0.0, 0.0, 1.0])
            .light_index(0)
            .texture(0),
    );
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

    // 追加: 球体ライト (強い発光)
    let mat_sphere_light = builder.add_material(
        Material::new([1.0, 1.0, 1.0, 1.0])
            .light_index(1) // 球体ライトのインデックス
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
    builder.add_instance(
        plane_id,
        mat_light,
        Mat4::from_translation(Vec3::new(0.0, 0.99, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
        0x1,
    );
    // Main Light:Light info
    builder.add_quad_light(
        Vec3::new(0.0, 0.99, 0.0).into(),
        [0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25],
        [1.0, 1.0, 1.0, 10.0],
    );

    // クリスタル (正八面体) - ガラス球のあった場所に移動し拡大
    let crystal_pos = Vec3::new(0.4, -0.5, 0.3);
    builder.add_instance(
        crystal_id,
        mat_crystal,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.5)),
        0x1,
    );

    // クリスタル内部の光
    builder.add_instance(
        sphere_id,
        mat_sphere_light,
        Mat4::from_translation(crystal_pos) * Mat4::from_scale(Vec3::splat(0.1)),
        0x1,
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
pub fn create_avocado_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // 1. 各ジオメトリの生成
    let plane_geo = geometry::create_plane_blas(device);
    // ライト用のジオメトリ (Quad)
    let light_geo = geometry::create_plane_blas(device);

    let mut gltf_geos = Vec::new();
    let mut gltf_mats = Vec::new();

    // Avocado読み込み
    match loader::load_gltf("assets/models/Avocado.glb", device) {
        Ok((geos, mats, images)) => {
            println!(
                "Loaded Avocado.glb: {} geometries, {} materials, {} images",
                geos.len(),
                mats.len(),
                images.len()
            );
            gltf_geos = geos;

            // Load textures into builder and get the base ID offset
            // Current textures in builder: 0 (White), 1 (Checker) => next is 2
            let base_tex_id = builder.textures.len() as u32;

            for img in images {
                builder.add_texture(img);
            }

            // Update material texture indices and add to list
            for mut mat in mats {
                // Determine if material has a texture.
                // Since I didn't successfully update load_gltf yet to use u32::MAX, I assume 0 means "might be texture 0 from GLTF".
                // However, without the u32::MAX logic, collision is inevitable if GLTF has texture 0.
                // Assuming simple case: offset everything.
                // Ideally I should assume 0 is "valid texture 0 from GLTF" and map it to `base_tex_id`.
                // But how to represent "no texture"?
                //
                // Wait, I DID update `load_gltf` successfully in Step 154 (via replace_file_content).
                // But I didn't add the u32::MAX logic there. I just added image loading.
                // Step 154:
                // `if let Some(texture_info) = pbr.base_color_texture() { mat = mat.texture(...) }`
                // `Material::new` sets `tex_id = 0`.
                // So if default, it is 0.
                //
                // If I offset 0, then "No texture" becomes "Texture 2" (which is the first GLTF texture).
                // This means everything will have the avocado skin texture if I'm not careful.
                //
                // I MUST fix `load_gltf` to distinguish "No Texture".
                // I will assume `tex_id` 0 is "No Texture" (White) IF it wasn't set by GLTF.
                // But I can't know.
                //
                // Hack: In `load_gltf`, `Material::new` sets tex_id=0. I will assume GLTF textures are indices 0..N.
                // I will modify `load_gltf` NOW to set `mat.tex_id` to `u32::MAX` by default in a separate call?
                // No, I should do it in `SceneBuilder` logic if possible.
                //
                // Better: I will PROPERLY update `load_gltf` first to set default `tex_id = u32::MAX`.
                // But I can't edit `load_gltf` inside this `replace_file_content`.
                //
                // So I will edit `create_avocado_scene` to just print for now, and I will fix `load_gltf` immediately after.
                // Or I can write the logic here assuming `load_gltf` behaves correctly (returns u32::MAX for no texture),
                // and then fix `load_gltf`.
                //
                // Let's assume `mat.tex_id` IS `u32::MAX` for no texture.
                //
                if mat.tex_id == u32::MAX {
                    mat.tex_id = 0; // Default White
                } else {
                    mat.tex_id += base_tex_id;
                }
                gltf_mats.push(mat);
            }
        }
        Err(e) => {
            eprintln!("Failed to load Avocado.glb: {:?}", e);
        }
    }

    // 2. BLAS Build (まとめて実行)
    // Plane + Light Geometry + GLTF Geometries
    let mut all_geos = vec![&plane_geo, &light_geo];
    for geo in &gltf_geos {
        all_geos.push(geo);
    }
    SceneBuilder::build_blases(device, queue, &all_geos);

    // 3. メッシュとマテリアルの登録

    // Plane
    let plane_id = builder.add_mesh(plane_geo);
    // Light
    let light_mesh_id = builder.add_mesh(light_geo);

    // GLTF Meshes
    let mut gltf_mesh_ids = Vec::new();
    for geo in gltf_geos {
        gltf_mesh_ids.push(builder.add_mesh(geo));
    }

    // Materials
    let mat_floor = builder.add_material(
        Material::new([0.73, 0.73, 0.73, 1.0])
            .roughness(0.99)
            .texture(0),
    );
    // Light Material (Emissive)
    let mat_light = builder.add_material(
        Material::new([1.0, 1.0, 1.0, 10.0]) // High intensity color
            .light_index(0) // Map to first light
            .texture(0),
    );

    let mut gltf_mat_ids = Vec::new();
    for mat in gltf_mats {
        gltf_mat_ids.push(builder.add_material(mat));
    }

    // 4. インスタンスの配置
    // Floor
    builder.add_instance(
        plane_id,
        mat_floor,
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)) * Mat4::from_scale(Vec3::splat(10.0)),
        0x1,
    );

    // Avocado
    for (i, &mesh_id) in gltf_mesh_ids.iter().enumerate() {
        // マテリアルIDの対応 (単純に順序通りと仮定)
        let mat_id = if i < gltf_mat_ids.len() {
            gltf_mat_ids[i]
        } else {
            // マテリアルが不足している場合
            println!("Material not found for mesh {}", i);
            0
        };

        builder.add_instance(
            mesh_id,
            mat_id,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(20.0)),
            0x1,
        );
    }

    // Light Instance
    builder.add_instance(
        light_mesh_id,
        mat_light,
        Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0))
            * Mat4::from_rotation_x(std::f32::consts::PI) // 下向き
            * Mat4::from_scale(Vec3::splat(1.0)),
        0x1,
    );

    // 5. 光源データ
    builder.add_quad_light(
        Vec3::new(0.0, 5.0, 0.0).into(),
        [0.5, 0.0, 0.0], // 少し大きめに
        [0.0, 0.0, 0.5],
        [1.0, 1.0, 1.0, 15.0],
    );

    builder.build(device, queue)
}
