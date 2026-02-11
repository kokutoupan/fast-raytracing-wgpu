use crate::scene;
use crate::wgpu_ctx::WgpuContext;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Reservoir {
    pub y: u32,           // Path seed
    pub w_sum: f32,       // Sum of weights
    pub m: u32,           // Number of samples seen
    pub w: f32,           // Generalized weight
    pub s_path: [f32; 3], // Path throughput
    pub p_hat: f32,       // Target density (Luminance)
}

impl Reservoir {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            y: 0,
            w_sum: 0.0,
            m: 0,
            w: 0.0,
            s_path: [0.0; 3],
            p_hat: 0.0,
        }
    }
}

pub struct RestirPass {
    pub pipeline: wgpu::ComputePipeline,
    pub reservoir_buffers: [wgpu::Buffer; 2],
    pub bind_groups0: [wgpu::BindGroup; 2],
    pub bind_group_layout1: wgpu::BindGroupLayout, // Textures
    pub bind_group2: wgpu::BindGroup,              // Reservoirs (Group 2)
    pub scene_info_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneInfo {
    light_count: u32,
    frame_count: u32,
    pad1: u32,
    pad2: u32,
}

impl RestirPass {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
        gbuffer_pos: &[wgpu::TextureView; 2],
        gbuffer_normal: &[wgpu::TextureView; 2],
        gbuffer_albedo: &[wgpu::TextureView; 2], // Double Buffered
        gbuffer_motion: &wgpu::TextureView,
        render_width: u32,
        render_height: u32,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/restir.wgsl"));

        // Bind Group 0: GBuffer, Camera, Lights, Scene Info (Same as before)
        let bgl0 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir BGL 0"),
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
                    // 2: Albedo (Current)
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
                    // 14: Prev Albedo (New)
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
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
                label: Some("Restir BGL 1 (Textures)"),
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

        // Bind Group 2: Reservoirs
        let bgl2 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir BGL 2 (Reservoirs)"),
                entries: &[
                    // 0: Prev (Read Only - though we use read_write storage in shader currently)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: Curr (Write)
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
                label: Some("Restir Pipeline Layout"),
                bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Restir Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Reservoirs
        let reservoir_size =
            (render_width * render_height * std::mem::size_of::<Reservoir>() as u32) as u64;
        let reservoir_buffers = [
            crate::wgpu_utils::create_buffer(
                &ctx.device,
                "Reservoir Buffer 0",
                reservoir_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            crate::wgpu_utils::create_buffer(
                &ctx.device,
                "Reservoir Buffer 1",
                reservoir_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
        ];

        let scene_info_buffer = crate::wgpu_utils::create_buffer_init(
            &ctx.device,
            "Restir Scene Info",
            &[SceneInfo {
                light_count: 0,
                frame_count: 0,
                pad1: 0,
                pad2: 0,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Bind Group 2 Fixed Flow:
        // Input: Buffer 1 (Previous Spatial Result)
        // Output: Buffer 0 (Current Temporal Result)
        let bind_group2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Restir BG2 (Fixed Flow)"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reservoir_buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: reservoir_buffers[0].as_entire_binding(),
                },
            ],
        });

        // Bind Group 0 Ping-Pong (G-Buffer)
        let create_bg0 = |label: &str,
                          curr_pos: &wgpu::TextureView,
                          curr_normal: &wgpu::TextureView,
                          curr_albedo: &wgpu::TextureView, // New
                          prev_pos: &wgpu::TextureView,
                          prev_normal: &wgpu::TextureView,
                          prev_albedo: &wgpu::TextureView| {
            // New
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
                        resource: wgpu::BindingResource::TextureView(curr_albedo), // Current Albedo
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
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::TextureView(prev_albedo), // Previous Albedo
                    },
                ],
            })
        };

        // Frame 0 logic: Write to Gbuffer 0.
        // Restir needs Curr=0, Prev=1.
        let bg0_0 = create_bg0(
            "Restir BG0 (Curr=0)",
            &gbuffer_pos[0],
            &gbuffer_normal[0],
            &gbuffer_albedo[0], // Curr Albedo
            &gbuffer_pos[1],
            &gbuffer_normal[1],
            &gbuffer_albedo[1], // Prev Albedo
        );

        // Frame 1 logic: Write to Gbuffer 1.
        // Restir needs Curr=1, Prev=0.
        let bg0_1 = create_bg0(
            "Restir BG0 (Curr=1)",
            &gbuffer_pos[1],
            &gbuffer_normal[1],
            &gbuffer_albedo[1], // Curr Albedo
            &gbuffer_pos[0],
            &gbuffer_normal[0],
            &gbuffer_albedo[0], // Prev Albedo
        );

        Self {
            pipeline,
            reservoir_buffers,
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
            label: Some("Restir Texture BG"),
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
            label: Some("Restir Pass"),
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
