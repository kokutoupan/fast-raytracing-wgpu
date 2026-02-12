use crate::wgpu_ctx::WgpuContext;

pub struct PostPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_groups: [wgpu::BindGroup; 2],
}

impl PostPass {
    pub fn new(
        ctx: &WgpuContext,
        raw_view: &wgpu::TextureView,
        accumulation_buffers: &[wgpu::Buffer; 2],
        pp_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        gbuffer_normal_views: &[wgpu::TextureView; 2],
        gbuffer_pos_views: &[wgpu::TextureView; 2],
        gbuffer_motion: &wgpu::TextureView, // New
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/post.wgsl"));

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Post Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Binding 3: G-Buffer Normal
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 4: G-Buffer Position
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 6: Motion (New)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 7: Accumulation Output (New, WriteOnly)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Post Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bgl],
                            immediate_size: 0,
                        },
                    )),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let create_bg = |label: &str,
                         normal: &wgpu::TextureView,
                         pos: &wgpu::TextureView,
                         history_buf: &wgpu::Buffer,
                         output_buf: &wgpu::Buffer| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(raw_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: history_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(pp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(pos),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(gbuffer_motion),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            })
        };

        // Frame 0: Read from Buf[1], Write to Buf[0]
        let bg0 = create_bg(
            "Post Bind Group 0",
            &gbuffer_normal_views[0],
            &gbuffer_pos_views[0],
            &accumulation_buffers[1], // History
            &accumulation_buffers[0], // Output
        );
        // Frame 1: Read from Buf[0], Write to Buf[1]
        let bg1 = create_bg(
            "Post Bind Group 1",
            &gbuffer_normal_views[1],
            &gbuffer_pos_views[1],
            &accumulation_buffers[0], // History
            &accumulation_buffers[1], // Output
        );

        Self {
            pipeline,
            bind_groups: [bg0, bg1],
        }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        frame_count: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Post Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        let idx = (frame_count % 2) as usize;
        cpass.set_bind_group(0, &self.bind_groups[idx], &[]);
        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
