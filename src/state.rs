use std::sync::Arc;
use wgpu::{
    Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    RequestAdapterOptions, util::DeviceExt,
};

use winit::window::Window;

// --- State 構造体定義 ---
pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,

    // レイトレ用リソース
    pub tlas: wgpu::Tlas,
    pub _blas: wgpu::Blas, // 参照を保持するために必要
    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,

    // 画面表示用 (Blit)
    pub storage_texture_view: wgpu::TextureView,
    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group: wgpu::BindGroup,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        // 1. 初期化 (前回の成功パターン)
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
                label: Some("RayTracing Device"),
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        // 2. 加速構造 (AS) の構築
        let vertices: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });

        let blas_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: 3,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("Triangle BLAS"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_size_desc.clone()],
            },
        );

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        // インデックス0に登録 (Identity行列)
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xff,
        ));

        // AS ビルド実行
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(
            Some(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &blas_size_desc,
                        vertex_buffer: &vertex_buf,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            Some(&tlas),
        );
        queue.submit(std::iter::once(encoder.finish()));

        // 3. パイプラインとシェーダーの設定

        // 作業用テクスチャ (Compute -> Render)
        let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Storage Texture"),
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
        let storage_texture_view = storage_texture.create_view(&Default::default());

        // シェーダーロード
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RayQuery Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader.wgsl"
            ))),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        // Compute パイプライン
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
            ],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
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
                    resource: wgpu::BindingResource::TextureView(&storage_texture_view),
                },
            ],
        });

        // Blit (表示用) パイプライン
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
            label: Some("Blit Pipeline"),
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
                    resource: wgpu::BindingResource::TextureView(&storage_texture_view),
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
            storage_texture_view,
            blit_pipeline,
            blit_bind_group,
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 1. Ray Query 計算パス
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

        // 2. 画面転送パス (Blit)
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
