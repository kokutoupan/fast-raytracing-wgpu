use crate::scene;
use crate::wgpu_ctx::WgpuContext;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Reservoir {
    pub y: u32, // Light Index (pad to vec4f in shader, but here u32?)
    // Shader: vec4f for y.
    // Wait, shader uses vec4f, so 16 bytes.
    // But in my restir.wgsl I defined it as struct with y: vec4f for some reason?
    // Let's check my write to restir.wgsl.
    // "y: vec4f" -> this was a mistake if I want to store just index.
    // But raytrace.rs has struct Reservoir { y: u32, ... }
    // Let's stick to matching raytrace.rs struct first, adjusting shader to match Rust is easier than adjusting Rust to match weird Shader.
    // raytrace.rs: y: u32, w_sum: f32, m: u32, w: f32
    // Total 16 bytes.
    // Shader Reservoir struct needs to match 16 bytes.
    // In restir.wgsl I wrote:
    // struct Reservoir {
    //    y: vec4f, ...
    // }
    // This is mismatch. I need to fix restir.wgsl as well.
    // For now, let's define Rust struct as 16 bytes.
    pub w_sum: f32,
    pub m: u32,
    pub w: f32,
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

pub struct RestirPass {
    pub pipeline: wgpu::ComputePipeline,
    pub reservoir_buffers: [wgpu::Buffer; 2],
    pub bind_groups: [wgpu::BindGroup; 2],
    pub bind_group0: wgpu::BindGroup,
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
        gbuffer_pos: &wgpu::TextureView,
        gbuffer_normal: &wgpu::TextureView,
        gbuffer_albedo: &wgpu::TextureView,
        gbuffer_motion: &wgpu::TextureView,
        render_width: u32,
        render_height: u32,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/restir.wgsl"));

        // Bind Group 0: GBuffer, Camera, Lights, Scene Info
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
                ],
            });

        // Bind Group 1: Reservoirs
        let bgl1 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Restir BGL 1"),
                entries: &[
                    // 0: Prev
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
                    // 1: Curr
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
                bind_group_layouts: &[&bgl0, &bgl1],
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

        // Bind Groups
        let create_bg = |label: &str, prev: &wgpu::Buffer, curr: &wgpu::Buffer| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: prev.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: curr.as_entire_binding(),
                    },
                ],
            })
        };

        let bg1_ping = create_bg(
            "Restir BG1 Ping",
            &reservoir_buffers[0],
            &reservoir_buffers[1],
        );
        let bg1_pong = create_bg(
            "Restir BG1 Pong",
            &reservoir_buffers[1],
            &reservoir_buffers[0],
        );

        // Common BG0 (will need to recreate if Gbuffer changes? No, views are stable unless resize)
        let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Restir BG0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(gbuffer_pos),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(gbuffer_normal),
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
            ],
        });

        // We store BG0 separately or just one? One BG0 is enough.
        // But we need 2 BG1s (ping pong).
        // Let's store them.
        // Wait, `bind_groups` in struct is [BindGroup; 2].
        // I should probably store BG0 in struct too, OR recreate it.
        // But struct definition has `bind_groups: [BindGroup; 2]`.
        // Let's redefine struct to hold what we need.
        // Actually, let's keep it simple: `bind_groups` hold (BG0, BG1_Ping) and (BG0, BG1_Pong)?
        // Pass execute needs to switch BG1. BG0 is constant.
        // RaytracePass had `bind_groups` as array of BGs where each invalidates everything? No.
        // RaytracePass: `set_bind_group(0, &self.bind_groups[(frame % 2)], ...)` -> It swapped Reservoir buffers which were in BG0.
        // Here I put Reservoirs in BG1.
        // So I need to set BG0 once, and flip BG1.
        // Let's reuse `bind_groups` to store BG1s.
        // And add `bind_group0` field.

        Self {
            pipeline,
            reservoir_buffers,
            bind_groups: [bg1_ping, bg1_pong],
            bind_group0: bg0,
            scene_info_buffer,
        }
    }

    // Need to execute
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &WgpuContext,
        width: u32,
        height: u32,
        frame_count: u32,
        light_count: u32,
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

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Restir Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group0, &[]);
        cpass.set_bind_group(1, &self.bind_groups[(frame_count % 2) as usize], &[]);
        cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
