use crate::scene;
use crate::wgpu_ctx::WgpuContext;
use crate::wgpu_utils::*;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlitParams {
    width: u32,
    height: u32,
    frame_count: u32,
    spp: u32,
}

pub struct Renderer {
    pub render_width: u32,
    pub render_height: u32,

    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,
    pub compute_bind_group_layout: wgpu::BindGroupLayout,

    pub blit1_pipeline: wgpu::RenderPipeline,
    pub blit1_bind_group: wgpu::BindGroup,
    pub blit1_bind_group_layout: wgpu::BindGroupLayout,

    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group: wgpu::BindGroup,
    pub blit_bind_group_layout: wgpu::BindGroupLayout,

    pub sampler: wgpu::Sampler,
    pub raw_raytrace_texture: wgpu::Texture,
    pub post_processed_texture: wgpu::Texture,
    pub accumulation_buffer: wgpu::Buffer,
    pub blit_params_buffer: wgpu::Buffer,
    pub frame_count: u32,
}

impl Renderer {
    pub fn aspect_ratio(&self) -> f32 {
        self.render_width as f32 / self.render_height as f32
    }
}

impl Renderer {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
    ) -> Self {
        // 固定の内部解像度 (例: 1280x720)
        let render_width = 1280;
        let render_height = 720;

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let blit1_shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("blit1.wgsl"));
        let blit_shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("blit.wgsl"));

        let accumulation_buffer = create_buffer(
            &ctx.device,
            "Accumulation Buffer",
            (render_width * render_height * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let raw_raytrace_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raw Raytrace Texture"),
            size: wgpu::Extent3d {
                width: render_width,
                height: render_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let raw_view = raw_raytrace_texture.create_view(&Default::default());

        let post_processed_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post Processed Texture"),
            size: wgpu::Extent3d {
                width: render_width,
                height: render_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pp_view = post_processed_texture.create_view(&Default::default());

        let blit_params = BlitParams {
            width: render_width,
            height: render_height,
            frame_count: 0,
            spp: 2,
        };
        let blit_params_buffer = create_buffer_init(
            &ctx.device,
            "Blit Params Buffer",
            &[blit_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let constants = [("MAX_DEPTH", 8.0), ("SPP", 2.0)];

        // --- Compute Pipeline Setup ---
        let compute_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
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
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&compute_bgl],
                            immediate_size: 0,
                        },
                    )),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                });

        let compute_bind_group = create_compute_bind_group(
            &ctx.device,
            &compute_bgl,
            scene_resources,
            &raw_view,
            camera_buffer,
            &accumulation_buffer,
        );

        let sampler = ctx
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        // --- Blit 1 Pipeline Setup ---
        let blit1_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit1 Bind Group Layout"),
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let blit1_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Blit1 Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&blit1_bgl],
                            immediate_size: 0,
                        },
                    )),
                    vertex: wgpu::VertexState {
                        module: &blit1_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &blit1_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
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

        let blit1_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit1 Bind Group"),
            layout: &blit1_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accumulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blit_params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Blit 2 Pipeline Setup ---
        let blit_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Blit Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&blit_bgl],
                            immediate_size: 0,
                        },
                    )),
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
                            format: ctx.config.format,
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

        let blit_bind_group = create_blit_bind_group(&ctx.device, &blit_bgl, &pp_view, &sampler);

        Self {
            render_width,
            render_height,
            compute_pipeline,
            compute_bind_group,
            compute_bind_group_layout: compute_bgl,
            blit1_pipeline,
            blit1_bind_group,
            blit1_bind_group_layout: blit1_bgl,
            blit_pipeline,
            blit_bind_group,
            blit_bind_group_layout: blit_bgl,
            sampler,
            raw_raytrace_texture,
            post_processed_texture,
            accumulation_buffer,
            blit_params_buffer,
            frame_count: 0,
        }
    }

    pub fn resize(
        &mut self,
        _ctx: &WgpuContext,
        _scene_resources: &scene::SceneResources,
        _camera_buffer: &wgpu::Buffer,
    ) {
        // 内部解像度は固定なため、ここでは何もしないか、あるいは必要に応じて拡大縮小ロジックを更新する
        // ウィンドウ資源に依存するバインドグループがあれば更新するが、今は pp_view に依存している
        self.frame_count = 0;
    }

    pub fn render(
        &mut self,
        ctx: &WgpuContext,
        view: &wgpu::TextureView,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Update Blit Params
        let blit_params = BlitParams {
            width: self.render_width,
            height: self.render_height,
            frame_count: self.frame_count,
            spp: 2,
        };
        ctx.queue.write_buffer(
            &self.blit_params_buffer,
            0,
            bytemuck::cast_slice(&[blit_params]),
        );

        // 1. Ray Tracing Pass (Compute)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Raytrace Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.render_width + 7) / 8, (self.render_height + 7) / 8, 1);
        }

        // 2. Blit 1 Pass (Accumulation & Gamma)
        {
            let pp_view = self.post_processed_texture.create_view(&Default::default());
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit1 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pp_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit1_pipeline);
            rpass.set_bind_group(0, &self.blit1_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // 3. Blit 2 Pass (Window Output)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;

        Ok(())
    }
}
