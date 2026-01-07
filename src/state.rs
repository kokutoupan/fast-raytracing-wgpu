use crate::camera::{CameraController, CameraUniform};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt; // cameraを使う

// GPUに送るマテリアルデータ (32バイト)
// color: 表面の色 (RGB), Aは予備
// emission: 発光の色と強さ (RGB), Aは強度(intensity)として使うと便利
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    color: [f32; 4],
    emission: [f32; 4],
}

// --- ヘルパー関数: 平面(Quad)のBLASを作成 ---
// 戻り値: (BLAS, 頂点バッファ(参照保持用), インデックスバッファ(参照保持用), サイズ記述子)
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

pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<winit::window::Window>,

    pub tlas: wgpu::Tlas,
    pub _blas: wgpu::Blas,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,

    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group: wgpu::BindGroup,

    // カメラ関連
    pub camera_buffer: wgpu::Buffer,
    pub camera_controller: CameraController,
}

impl State {
    pub async fn new(window: winit::window::Window) -> Self {
        let window = Arc::new(window);

        // 1. 初期化 (前回と同じ)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::from_env_or_default(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .unwrap();
        let config = surface
            .get_default_config(
                &adapter,
                window.inner_size().width,
                window.inner_size().height,
            )
            .unwrap();
        surface.configure(&device, &config);

        // 2. AS構築 (ヘルパー関数使用)
        let (blas, v_buf, i_buf, blas_desc) = create_plane_blas(&device);

        // TLAS作成 (Cornell Boxの配置)
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

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(
            Some(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &blas_desc,
                        vertex_buffer: &v_buf,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: Some(&i_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
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
            // Storage Buffer (ReadOnly) として使う
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 3. カメラ初期化
        let camera_controller = CameraController::new();
        let camera_uniform =
            camera_controller.build_uniform(config.width as f32 / config.height as f32);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 4. パイプライン (Binding 2 にカメラを追加)
        let storage_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let storage_view = storage_tex.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("blit.wgsl"));

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // カメラ用
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // マテリアル用
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&compute_bgl],
                    immediate_size: 0,
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        });

        // Blit (省略なし)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&blit_bgl],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            device,
            queue,
            surface,
            config,
            window,
            tlas,
            _blas: blas,
            compute_pipeline,
            compute_bind_group,
            blit_pipeline,
            blit_bind_group,
            camera_buffer,
            camera_controller,
        }
    }

    // 入力イベントをカメラに渡す
    pub fn input(&mut self, event: &winit::event::WindowEvent) {
        self.camera_controller.process_events(event);
    }

    // フレームごとの更新処理
    pub fn update(&mut self) {
        self.camera_controller.update_camera();
        let uniform = self
            .camera_controller
            .build_uniform(self.config.width as f32 / self.config.height as f32);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.compute_bind_group), &[]);
            cpass.dispatch_workgroups(
                self.config.width.div_ceil(8),
                self.config.height.div_ceil(8),
                1,
            );
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, Some(&self.blit_bind_group), &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
