use crate::scene;
use crate::wgpu_ctx::WgpuContext;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Reservoir {
    pub y: u32,     // Light Index
    pub w_sum: f32, // Sum of weights
    pub m: u32,     // Number of samples seen
    pub w: f32,     // Generalized weight
}

impl Reservoir {
    pub fn new() -> Self {
        Self {
            y: 0,
            w_sum: 0.0,
            m: 0,
            w: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneInfo {
    light_count: u32,
    frame_count: u32,
    pad1: u32,
    pad2: u32,
}

pub struct RestirSpatialPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_groups0: [wgpu::BindGroup; 2],
    pub bind_group_layout1: wgpu::BindGroupLayout, // Textures
    pub bind_group2: wgpu::BindGroup,              // Reservoirs (Group 2)
    pub scene_info_buffer: wgpu::Buffer,
}

impl RestirSpatialPass {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
        gbuffer_pos: &[wgpu::TextureView; 2],
        gbuffer_normal: &[wgpu::TextureView; 2],
        gbuffer_albedo: &wgpu::TextureView,
        gbuffer_motion: &wgpu::TextureView,
        reservoirs_temporal: &wgpu::Buffer, // Read (Input)
        reservoirs_spatial: &wgpu::Buffer,  // Write (Output)
        render_width: u32,
        render_height: u32,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/restir_spatial.wgsl"));

        // Bind Group 0: GBuffer, Camera, Lights, Scene Info, Geometry, Prev GBuffer
        let bgl0 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir Spatial BGL 0"),
                entries: &[
                    // 0: Pos
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
                    // 1: Normal
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: Albedo
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: Motion
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
                    // 4: Camera
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: Lights
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
                    // 6: Scene Info
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 7: TLAS
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    // 8: Materials
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 9: Attributes
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 10: Indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 11: MeshInfos
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 12: Prev Pos
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 13: Prev Normal
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Bind Group 1: Textures
        let bgl1 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir Spatial BGL 1 (Textures)"),
                entries: &[
                    // 0: Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 1: Textures
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

        // Bind Group 2: Reservoirs (Input -> Output)
        let bgl2 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir Spatial BGL 2"),
                entries: &[
                    // 0: Input (Temporal Result)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: Output (Spatial Result)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Restir Spatial Pipeline Layout"),
                bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Restir Spatial Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let scene_info_buffer = crate::wgpu_utils::create_buffer_init(
            &ctx.device,
            "Restir Spatial Scene Info",
            &[SceneInfo {
                light_count: 0,
                frame_count: 0,
                pad1: 0,
                pad2: 0,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Bind Group 2: Fixed Flow (Buf0 -> Buf1)
        let bind_group2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Restir Spatial BG2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reservoirs_temporal.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: reservoirs_spatial.as_entire_binding(),
                },
            ],
        });

        // Bind Group 0 Ping-Pong (G-Buffer)
        let create_bg0 = |label: &str,
                          curr_pos: &wgpu::TextureView,
                          curr_normal: &wgpu::TextureView,
                          prev_pos: &wgpu::TextureView,
                          prev_normal: &wgpu::TextureView| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(curr_pos),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(curr_normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(gbuffer_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(gbuffer_motion),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: scene_resources.light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: scene_info_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::AccelerationStructure(
                            &scene_resources.tlas,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: scene_resources.material_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: scene_resources.global_attribute_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: scene_resources.global_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: scene_resources.mesh_info_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(prev_pos),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::TextureView(prev_normal),
                    },
                ],
            })
        };

        // Frame 0 logic: Write to Gbuffer 0.
        // Restir needs Curr=0, Prev=1.
        let bg0_0 = create_bg0(
            "Restir Spatial BG0 (Curr=0)",
            &gbuffer_pos[0],
            &gbuffer_normal[0],
            &gbuffer_pos[1],
            &gbuffer_normal[1],
        );

        // Frame 1 logic: Write to Gbuffer 1.
        // Restir needs Curr=1, Prev=0.
        let bg0_1 = create_bg0(
            "Restir Spatial BG0 (Curr=1)",
            &gbuffer_pos[1],
            &gbuffer_normal[1],
            &gbuffer_pos[0],
            &gbuffer_normal[0],
        );

        Self {
            pipeline,
            bind_groups0: [bg0_0, bg0_1],
            bind_group_layout1: bgl1,
            bind_group2,
            scene_info_buffer,
        }
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &WgpuContext,
        width: u32,
        height: u32,
        frame_count: u32,
        light_count: u32,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        // Update Scene Info
        ctx.queue.write_buffer(
            &self.scene_info_buffer,
            0,
            bytemuck::cast_slice(&[SceneInfo {
                light_count,
                frame_count,
                pad1: 0,
                pad2: 0,
            }]),
        );

        let bind_group1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Restir Spatial Texture BG"),
            layout: &self.bind_group_layout1,
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

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Restir Spatial Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);

        // idx = frame_count % 2
        let idx = (frame_count % 2) as usize;

        cpass.set_bind_group(0, &self.bind_groups0[idx], &[]);
        cpass.set_bind_group(1, &bind_group1, &[]);
        cpass.set_bind_group(2, &self.bind_group2, &[]);

        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
