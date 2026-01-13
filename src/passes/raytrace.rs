use crate::scene;
use crate::wgpu_ctx::WgpuContext;

pub struct RaytracePass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group0: wgpu::BindGroup,
    pub bind_group1: wgpu::BindGroup,
}

impl RaytracePass {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
        raw_view: &wgpu::TextureView,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/raytrace.wgsl"));

        let bgl0 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raytrace Bind Group Layout 0"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });

        let bgl1 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raytrace Bind Group Layout 1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let constants = [("MAX_DEPTH", 8.0), ("SPP", 2.0)];
        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Raytrace Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bgl0, &bgl1],
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

        let bind_group0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raytrace Bind Group 0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::AccelerationStructure(&scene_resources.tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene_resources.material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene_resources.global_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scene_resources.global_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scene_resources.mesh_info_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raytrace Bind Group 1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
            ],
        });

        Self {
            pipeline,
            bind_group0,
            bind_group1,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Raytrace Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group0, &[]);
        cpass.set_bind_group(1, &self.bind_group1, &[]);
        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
