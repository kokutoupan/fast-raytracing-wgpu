use crate::passes::{BlitPass, GBufferPass, PostPass, RestirPass, ShadePass};
use crate::scene;
use crate::wgpu_ctx::WgpuContext;
use crate::wgpu_utils::*;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PostParams {
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub spp: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlitParams {
    pub scale: [f32; 2],
    pub _padding: [f32; 2],
}

#[allow(dead_code)]
pub struct RenderTargets {
    pub width: u32,
    pub height: u32,

    pub raw_raytrace_texture: wgpu::Texture,
    pub post_processed_texture: wgpu::Texture,
    pub accumulation_buffer: wgpu::Buffer,

    // --- G-Buffer Textures ---
    pub gbuffer_pos: wgpu::Texture, // World Position (xyz) + Linear Depth (w)
    pub gbuffer_normal: wgpu::Texture, // World Normal (xyz) + Roughness (w)
    pub gbuffer_albedo: wgpu::Texture, // Albedo (rgb) + Metallic/MatID (w)
    pub gbuffer_motion: wgpu::Texture, // Motion Vector (xy)

    // Views (convenience)
    pub raw_view: wgpu::TextureView,
    pub pp_view: wgpu::TextureView,

    // Views (Binding用)
    pub gbuffer_pos_view: wgpu::TextureView,
    pub gbuffer_normal_view: wgpu::TextureView,
    pub gbuffer_albedo_view: wgpu::TextureView,
    pub gbuffer_motion_view: wgpu::TextureView,
}

impl RenderTargets {
    pub fn new(ctx: &WgpuContext, width: u32, height: u32) -> Self {
        let accumulation_buffer = create_buffer(
            &ctx.device,
            "Accumulation Buffer",
            (width * height * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let raw_raytrace_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raw Raytrace Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let raw_view = raw_raytrace_texture.create_view(&Default::default());

        let post_processed_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post Processed Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // --- Helper for creating storage textures ---
        let create_storage_tex = |label: &str, format: wgpu::TextureFormat| {
            ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };

        // G-Buffer Textures
        let gbuffer_pos = create_storage_tex("GBuffer Position", wgpu::TextureFormat::Rgba32Float);
        let gbuffer_normal = create_storage_tex("GBuffer Normal", wgpu::TextureFormat::Rgba32Float);
        let gbuffer_albedo = create_storage_tex("GBuffer Albedo", wgpu::TextureFormat::Rgba8Unorm);
        let gbuffer_motion = create_storage_tex("GBuffer Motion", wgpu::TextureFormat::Rg32Float);

        // Views
        let gbuffer_pos_view = gbuffer_pos.create_view(&Default::default());
        let gbuffer_normal_view = gbuffer_normal.create_view(&Default::default());
        let gbuffer_albedo_view = gbuffer_albedo.create_view(&Default::default());
        let gbuffer_motion_view = gbuffer_motion.create_view(&Default::default());

        let pp_view = post_processed_texture.create_view(&Default::default());

        Self {
            width,
            height,
            raw_raytrace_texture,
            post_processed_texture,
            accumulation_buffer,
            gbuffer_pos,
            gbuffer_normal,
            gbuffer_albedo,
            gbuffer_motion,
            raw_view,
            pp_view,
            gbuffer_pos_view,
            gbuffer_normal_view,
            gbuffer_albedo_view,
            gbuffer_motion_view,
        }
    }
}

#[allow(dead_code)]
pub struct Renderer {
    pub render_width: u32,
    pub render_height: u32,
    pub window_width: u32,
    pub window_height: u32,

    // Passes
    gbuffer_pass: GBufferPass,
    restir_pass: RestirPass,
    shade_pass: ShadePass,
    post_pass: PostPass,
    blit_pass: BlitPass,

    // Render Targets (Screen Size Dependent)
    pub targets: RenderTargets,

    // Shared Resources (Screen Size Independent)
    pub post_params_buffer: wgpu::Buffer,
    pub blit_params_buffer: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    pub texture_array: wgpu::Texture,

    pub frame_count: u32,
}

impl Renderer {
    pub fn aspect_ratio(&self) -> f32 {
        self.render_width as f32 / self.render_height as f32
    }

    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
        render_width: u32,
        render_height: u32,
    ) -> Self {
        // --- Render Targets ---
        let targets = RenderTargets::new(ctx, render_width, render_height);

        let post_params_buffer = create_buffer_init(
            &ctx.device,
            "Post Params Buffer",
            &[PostParams {
                width: render_width,
                height: render_height,
                frame_count: 0,
                spp: 2,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let blit_params_buffer = create_buffer_init(
            &ctx.device,
            "Blit Params Buffer",
            &[BlitParams {
                scale: [1.0, 1.0],
                _padding: [0.0; 2],
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // --- Texture Array (0: White, 1: Checker) ---
        let tex_dim = 512;
        let texture_size = wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: 2,
        };
        let texture_array = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Layer 0: White
        let white_data = generate_white_texture_data(tex_dim);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &white_data,
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

        // Layer 1: Checker
        let checker_data = generate_checkerboard_texture_data(tex_dim, 8);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                aspect: wgpu::TextureAspect::All,
            },
            &checker_data,
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

        let _texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });

        // --- Passes ---
        let gbuffer_pass = GBufferPass::new(
            ctx,
            scene_resources,
            camera_buffer,
            &targets.gbuffer_pos_view,
            &targets.gbuffer_normal_view,
            &targets.gbuffer_albedo_view,
            &targets.gbuffer_motion_view,
        );

        let restir_pass = RestirPass::new(
            ctx,
            scene_resources,
            camera_buffer,
            &targets.gbuffer_pos_view,
            &targets.gbuffer_normal_view,
            &targets.gbuffer_albedo_view,
            &targets.gbuffer_motion_view,
            render_width,
            render_height,
        );

        let post_pass = PostPass::new(
            ctx,
            &targets.raw_view,
            &targets.accumulation_buffer,
            &targets.pp_view,
            &post_params_buffer,
        );

        let shade_pass = ShadePass::new(ctx);

        let blit_pass = BlitPass::new(ctx, &targets.pp_view, &sampler, &blit_params_buffer);

        Self {
            render_width,
            render_height,
            window_width: ctx.config.width,
            window_height: ctx.config.height,
            gbuffer_pass,
            restir_pass,
            shade_pass,
            post_pass,
            blit_pass,
            targets,
            post_params_buffer,
            blit_params_buffer,
            sampler,
            texture_array,
            frame_count: 0,
        }
    }

    pub fn resize(
        &mut self,
        ctx: &WgpuContext,
        _scene_resources: &scene::SceneResources,
        _camera_buffer: &wgpu::Buffer,
    ) {
        self.window_width = ctx.config.width;
        self.window_height = ctx.config.height;
        self.frame_count = 0;
    }

    pub fn render(
        &mut self,
        ctx: &WgpuContext,
        view: &wgpu::TextureView,
        camera_buffer: &wgpu::Buffer,
        light_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        mesh_info_buffer: &wgpu::Buffer,
        tlas: &wgpu::Tlas,
        light_count: u32,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Update Post Params
        ctx.queue.write_buffer(
            &self.post_params_buffer,
            0,
            bytemuck::cast_slice(&[PostParams {
                width: self.render_width,
                height: self.render_height,
                frame_count: self.frame_count,
                spp: 2,
            }]),
        );

        // Update Blit Scale for aspect ratio correction
        let render_ar = self.aspect_ratio();
        let window_ar = self.window_width as f32 / self.window_height as f32;
        let mut scale = [1.0f32, 1.0f32];
        if window_ar > render_ar {
            scale[0] = render_ar / window_ar;
        } else {
            scale[1] = window_ar / render_ar;
        }
        ctx.queue.write_buffer(
            &self.blit_params_buffer,
            0,
            bytemuck::cast_slice(&[BlitParams {
                scale,
                _padding: [0.0; 2],
            }]),
        );

        // 0. G-Buffer Pass
        self.gbuffer_pass
            .execute(&mut encoder, self.render_width, self.render_height);

        // ---------------------------------------------------------------------
        // DEBUG: G-Buffer Visualization
        // 0: Shaded, 1: Pos (Float), 2: Normal (Float), 3: Albedo (Unorm), 4: Motion
        let debug_mode = 0;

        if debug_mode == 0 {
            // 1. ReSTIR Pass (Compute reservoirs)
            self.restir_pass.execute(
                &mut encoder,
                ctx,
                self.render_width,
                self.render_height,
                self.frame_count,
                light_count,
            );

            // Determine which reservoir buffer is the *output* of the current frame
            // In restir.wgsl:
            // @group(1) @binding(1) var<storage, read_write> curr_reservoirs: array<Reservoir>;
            // In RestirPass::execute:
            // cpass.set_bind_group(1, &self.bind_groups[(frame_count % 2)], ...);
            // If frame % 2 == 0 (Ping BG): Binding 1 is Buf[1]. So Output is Buf[1].
            // If frame % 2 == 1 (Pong BG): Binding 1 is Buf[0]. So Output is Buf[0].
            // Wait, logic check:
            // Ping (Frame 0): Prev=0, Curr=1. Output=1.
            // Pong (Frame 1): Prev=1, Curr=0. Output=0.
            // Correct.
            let current_reservoir_buffer = if self.frame_count % 2 == 0 {
                &self.restir_pass.reservoir_buffers[1]
            } else {
                &self.restir_pass.reservoir_buffers[0]
            };

            // 2. Shade Pass (Use GBuffer + Reservoirs)
            self.shade_pass.execute(
                &mut encoder,
                ctx,
                self.render_width,
                self.render_height,
                &self.targets.gbuffer_pos_view,
                &self.targets.gbuffer_normal_view,
                &self.targets.gbuffer_albedo_view,
                &self.targets.raw_view, // Using raw_view as output for now (Shade pass writes here)
                camera_buffer,
                light_buffer,
                current_reservoir_buffer,
                tlas,
                material_buffer,
                vertex_buffer,
                index_buffer,
                mesh_info_buffer,
            );

            // 3. Post Pass (Tonemap + Accumulate?)
            // ShadePass writes to raw_view.
            // PostPass reads raw_view, writes to pp_view.
            self.post_pass
                .execute(&mut encoder, self.render_width, self.render_height);
        } else if debug_mode == 1 || debug_mode == 2 || debug_mode == 4 {
            // Visualize Float Texture (Pos / Normal / Motion) -> PostPass (Tonemap)
            let src_tex = if debug_mode == 1 {
                &self.targets.gbuffer_pos
            } else if debug_mode == 2 {
                &self.targets.gbuffer_normal
            } else {
                &self.targets.gbuffer_motion
            };

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: src_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.raw_raytrace_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.render_width,
                    height: self.render_height,
                    depth_or_array_layers: 1,
                },
            );
            self.post_pass
                .execute(&mut encoder, self.render_width, self.render_height);
        } else {
            // Visualize Albedo (Unorm) -> Skip PostPass
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.gbuffer_albedo,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.post_processed_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.render_width,
                    height: self.render_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        // ---------------------------------------------------------------------

        // 4. Blit Pass
        self.blit_pass.execute(&mut encoder, view);

        ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;

        Ok(())
    }
}
