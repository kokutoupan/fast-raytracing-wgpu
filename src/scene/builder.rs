use super::material::Material;
use super::resources::{MeshInfo, SceneResources};
use crate::geometry::{self, VertexAttributes};
use crate::scene::light::LightUniform;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use image::DynamicImage;

pub struct SceneBuilder {
    pub materials: Vec<Material>,
    pub attributes: Vec<VertexAttributes>,
    pub indices: Vec<u32>,
    pub mesh_infos: Vec<MeshInfo>,
    pub instances: Vec<Option<wgpu::TlasInstance>>,
    pub blases: Vec<wgpu::Blas>,
    pub lights: Vec<LightUniform>,
    pub textures: Vec<DynamicImage>, // Added
}

impl SceneBuilder {
    pub fn new() -> Self {
        let mut builder = Self {
            materials: Vec::new(),
            attributes: Vec::new(),
            indices: Vec::new(),
            mesh_infos: Vec::new(),
            instances: Vec::new(),
            blases: Vec::new(),
            lights: Vec::new(),
            textures: Vec::new(),
        };
        // Add default textures
        builder.add_default_textures();
        builder
    }

    fn add_default_textures(&mut self) {
        use image::{ImageBuffer, Rgba};
        // 0: White
        let white = DynamicImage::ImageRgba8(ImageBuffer::from_fn(512, 512, |_, _| {
            Rgba([255, 255, 255, 255])
        }));
        self.textures.push(white);

        // 1: Checker
        let checker = DynamicImage::ImageRgba8(ImageBuffer::from_fn(512, 512, |x, y| {
            let check = ((x / 64) + (y / 64)) % 2 == 0;
            if check {
                Rgba([255, 255, 255, 255])
            } else {
                Rgba([0, 0, 0, 255])
            }
        }));
        self.textures.push(checker);

        // 2: Flat Normal (128, 128, 255)
        let flat_normal = DynamicImage::ImageRgba8(ImageBuffer::from_fn(512, 512, |_, _| {
            Rgba([128, 128, 255, 255])
        }));
        self.textures.push(flat_normal);

        // 3: Black (0, 0, 0) - For Zero Emissive / Zero Roughness etc.
        let black =
            DynamicImage::ImageRgba8(ImageBuffer::from_fn(512, 512, |_, _| Rgba([0, 0, 0, 255])));
        self.textures.push(black);
    }

    pub fn add_texture(&mut self, texture: DynamicImage) -> u32 {
        let id = self.textures.len() as u32;
        // Ensure RGBA8
        let rgba = if texture.color() != image::ColorType::Rgba8 {
            DynamicImage::ImageRgba8(texture.to_rgba8())
        } else {
            texture
        };
        self.textures.push(rgba);
        id
    }

    pub fn add_material(&mut self, mat: Material) -> u32 {
        let id = self.materials.len() as u32;
        self.materials.push(mat);
        id
    }

    pub fn add_mesh(&mut self, geo: geometry::Geometry) -> u32 {
        let id = self.mesh_infos.len() as u32;

        let v_offset = self.attributes.len() as u32; // Use attributes len
        let i_offset = self.indices.len() as u32;

        self.attributes.extend_from_slice(&geo.attributes);
        self.indices.extend_from_slice(&geo.indices);

        self.mesh_infos.push(MeshInfo {
            vertex_offset: v_offset,
            index_offset: i_offset,
            pad: [0; 2],
        });

        self.blases.push(geo.blas);

        id
    }

    pub fn build_blases(device: &wgpu::Device, queue: &wgpu::Queue, geos: &[&geometry::Geometry]) {
        let mut encoder = device.create_command_encoder(&Default::default());

        for geo in geos {
            let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp BLAS Vertex Buffer"),
                contents: bytemuck::cast_slice(&geo.positions), // Use positions!
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp BLAS Index Buffer"),
                contents: bytemuck::cast_slice(&geo.indices),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });

            encoder.build_acceleration_structures(
                std::iter::once(&wgpu::BlasBuildEntry {
                    blas: &geo.blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &geo.desc,
                            vertex_buffer: &v_buf,
                            first_vertex: 0,
                            vertex_stride: 16, // vec4f position
                            index_buffer: Some(&i_buf),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }),
                None,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn add_instance(&mut self, mesh_id: u32, mat_id: u32, transform: Mat4, _mask: u8) {
        let blas = &self.blases[mesh_id as usize];
        let affine = transform.transpose().to_cols_array();
        let instance_id = (mesh_id << 16) | mat_id;

        let instance =
            wgpu::TlasInstance::new(blas, affine[..12].try_into().unwrap(), instance_id, 0xff);
        self.instances.push(Some(instance));
    }

    pub fn add_gltf_materials(
        &mut self,
        materials: Vec<Material>,
        images: Vec<DynamicImage>,
    ) -> Vec<u32> {
        // Load textures into builder and get the base ID offset
        let base_tex_id = self.textures.len() as u32;

        for img in images {
            self.add_texture(img);
        }

        let mut gltf_mat_ids = Vec::new();
        for mut mat in materials {
            // Update material texture indices and add to list
            // Base Color
            let tex_id = mat.tex_id();
            if tex_id == 0xFFFF {
                mat = mat.texture(0); // Default White
            } else {
                mat = mat.texture(tex_id + base_tex_id);
            }

            // Normal
            let normal_tex_id = mat.normal_tex_id();
            if normal_tex_id == 0xFFFF {
                mat = mat.normal_texture(2); // Default Flat Normal
            } else {
                mat = mat.normal_texture(normal_tex_id + base_tex_id);
            }

            // Occlusion
            let occlusion_tex_id = mat.occlusion_tex_id();
            if occlusion_tex_id == 0xFFFF {
                mat = mat.occlusion_texture(0); // Default White (No occlusion)
            } else {
                mat = mat.occlusion_texture(occlusion_tex_id + base_tex_id);
            }

            // Emissive
            let emissive_tex_id = mat.emissive_tex_id();
            if emissive_tex_id == 0xFFFF {
                mat = mat.emissive_texture(3); // Default Black (No emission)
            } else {
                mat = mat.emissive_texture(emissive_tex_id + base_tex_id);
            }

            // Metallic Roughness
            let mr_tex_id = mat.metallic_roughness_tex_id();
            if mr_tex_id != 0xFFFF {
                mat = mat.metallic_roughness_texture(mr_tex_id + base_tex_id);
            }

            gltf_mat_ids.push(self.add_material(mat));
        }
        gltf_mat_ids
    }

    pub fn add_gltf_meshes(&mut self, geometries: Vec<geometry::Geometry>) -> Vec<u32> {
        let mut ids = Vec::new();
        for geo in geometries {
            ids.push(self.add_mesh(geo));
        }
        ids
    }

    pub fn add_gltf_instances(
        &mut self,
        mesh_ids: &[u32],
        mat_ids: &[u32],
        material_indices: &[usize],
        transform: Mat4,
    ) {
        for (i, &mesh_id) in mesh_ids.iter().enumerate() {
            let mat_index = if i < material_indices.len() {
                material_indices[i]
            } else {
                eprintln!("Material index out of bounds for mesh {}", i);
                0
            };

            let mat_id = if mat_index < mat_ids.len() {
                mat_ids[mat_index]
            } else {
                eprintln!("Material ID out of bounds for mesh {}", i);
                0 // Default fallback
            };

            self.add_instance(mesh_id, mat_id, transform, 0x1);
        }
    }

    pub fn register_quad_light(
        &mut self,
        mesh_id: u32,
        transform: Mat4,
        color: [f32; 3],
        intensity: f32,
    ) {
        // 1. Calculate emission
        let emission_factor = [
            color[0] * intensity,
            color[1] * intensity,
            color[2] * intensity,
        ];

        // 2. Create Material
        let mat_id = self.add_material(
            Material::new([1.0, 1.0, 1.0, 1.0])
                .light_index(self.lights.len() as i32)
                .emissive_factor(emission_factor)
                .texture(0), // Default White
        );

        // 3. Add Instance
        self.add_instance(mesh_id, mat_id, transform, 0x1);

        // 4. Register Light Uniform (NEE)
        let position: [f32; 3] = transform.w_axis.truncate().into();

        // Plane BLAS is 1x1 (-0.5 to 0.5), so "radius" is 0.5
        let u = (transform.transform_vector3(Vec3::X) * 0.5).into();
        let v = (transform.transform_vector3(Vec3::NEG_Z) * 0.5).into();

        self.add_quad_light(position, u, v, [color[0], color[1], color[2], intensity]);
    }

    pub fn register_sphere_light(
        &mut self,
        mesh_id: u32,
        transform: Mat4,
        color: [f32; 3],
        intensity: f32,
    ) {
        let emission_factor = [
            color[0] * intensity,
            color[1] * intensity,
            color[2] * intensity,
        ];

        let mat_id = self.add_material(
            Material::new([1.0, 1.0, 1.0, 1.0])
                .light_index(self.lights.len() as i32)
                .emissive_factor(emission_factor)
                .texture(0),
        );

        self.add_instance(mesh_id, mat_id, transform, 0x1);

        let position: [f32; 3] = transform.w_axis.truncate().into();
        // Sphere BLAS is radius 0.5 (diameter 1.0)
        let scale = transform.transform_vector3(Vec3::X).length();
        let real_radius = scale * 0.5;

        self.add_sphere_light(
            position,
            real_radius,
            [color[0], color[1], color[2], intensity],
        );
    }

    /// 矩形ライトを追加する
    /// - position: 中心座標
    /// - u: 中心から「右端」へのベクトル（向きと長さ = 幅の半分）
    /// - v: 中心から「上端」へのベクトル（向きと長さ = 高さの半分）
    /// - emission: 発光色 * 強度
    pub fn add_quad_light(
        &mut self,
        position: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        emission: [f32; 4],
    ) {
        // 面積計算:
        // u, v は「半径」相当なので、辺の長さは 2|u|, 2|v|
        // 平行四辺形の面積 = |(2u) x (2v)| = 4 * |u x v|
        let u_vec = Vec3::from(u);
        let v_vec = Vec3::from(v);
        let area = u_vec.cross(v_vec).length() * 4.0;

        self.lights.push(LightUniform {
            position,
            type_: 0, // Quad
            u,        // そのまま格納
            area,
            v, // そのまま格納
            pad: 0,
            emission,
        });
    }

    // 球体ライトを追加するヘルパー
    pub fn add_sphere_light(&mut self, center: [f32; 3], radius: f32, emission: [f32; 4]) {
        let area = 4.0 * std::f32::consts::PI * radius * radius;
        self.lights.push(LightUniform {
            position: center,
            type_: 1, // Sphere
            u: [0.0; 3],
            area,
            v: [radius, 0.0, 0.0], // v.x に半径を入れておくルールにする
            pad: 0,
            emission,
        });
    }

    pub fn build(mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
        let global_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Global Attribute Buffer"),
                contents: bytemuck::cast_slice(&self.attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let global_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Info Buffer"),
            contents: bytemuck::cast_slice(&self.mesh_infos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: bytemuck::cast_slice(&self.materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: self.instances.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        // Indexing into TLAS to set instances
        for (i, instance) in self.instances.drain(..).enumerate() {
            tlas[i] = instance;
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(None, Some(&tlas));
        queue.submit(std::iter::once(encoder.finish()));

        // Light Buffer作成
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&self.lights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // --- Texture Array Building ---
        let tex_dim = 512;
        let texture_size = wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: self.textures.len() as u32,
        };
        let texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Texture Array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (i, img) in self.textures.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture_array,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                img.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_dim * 4),
                    rows_per_image: Some(tex_dim),
                },
                wgpu::Extent3d {
                    width: tex_dim,
                    height: tex_dim,
                    depth_or_array_layers: 1,
                },
            );
        }

        let texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Scene Texture Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(self.textures.len() as u32),
            ..Default::default()
        });

        SceneResources {
            tlas,
            global_attribute_buffer,
            global_index_buffer,
            mesh_info_buffer,
            blases: self.blases,
            material_buffer,
            light_buffer,
            num_lights: self.lights.len() as u32,
            texture_view,
        }
    }
}
