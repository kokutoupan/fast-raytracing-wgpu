use crate::passes::{BlitPass, GBufferPass, PostPass, RestirPass, RestirSpatialPass};
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
    pub gbuffer_pos: [wgpu::Texture; 2],
    pub gbuffer_normal: [wgpu::Texture; 2],
    pub gbuffer_albedo: [wgpu::Texture; 2], // Double buffered
    pub gbuffer_motion: wgpu::Texture,      // Motion Vector (xy)

    // Views (convenience)
    pub raw_view: wgpu::TextureView,
    pub pp_view: wgpu::TextureView,

    // Views (Binding用)
    pub gbuffer_pos_view: [wgpu::TextureView; 2],
    pub gbuffer_normal_view: [wgpu::TextureView; 2],
    pub gbuffer_albedo_view: [wgpu::TextureView; 2],
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
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST, // Added COPY_DST for consistency with provided snippet
                view_formats: &[],
            })
        };

        // G-Buffer Textures
        let gbuffer_pos = [
            create_storage_tex("GBuffer Position 0", wgpu::TextureFormat::Rgba32Float),
            create_storage_tex("GBuffer Position 1", wgpu::TextureFormat::Rgba32Float),
        ];
        let gbuffer_normal = [
            create_storage_tex("GBuffer Normal 0", wgpu::TextureFormat::Rgba32Float),
            create_storage_tex("GBuffer Normal 1", wgpu::TextureFormat::Rgba32Float),
        ];
        let gbuffer_albedo = [
            create_storage_tex("GBuffer Albedo 0", wgpu::TextureFormat::Rgba8Unorm),
            create_storage_tex("GBuffer Albedo 1", wgpu::TextureFormat::Rgba8Unorm),
        ];
        let gbuffer_motion = create_storage_tex("GBuffer Motion", wgpu::TextureFormat::Rg32Float);

        // Views
        let gbuffer_pos_view = [
            gbuffer_pos[0].create_view(&Default::default()),
            gbuffer_pos[1].create_view(&Default::default()),
        ];
        let gbuffer_normal_view = [
            gbuffer_normal[0].create_view(&Default::default()),
            gbuffer_normal[1].create_view(&Default::default()),
        ];
        let gbuffer_albedo_view = [
            gbuffer_albedo[0].create_view(&Default::default()),
            gbuffer_albedo[1].create_view(&Default::default()),
        ];
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
    restir_spatial_pass: RestirSpatialPass,
    // shade_pass: ShadePass, // Removed
    post_pass: PostPass,
    blit_pass: BlitPass,

    // Render Targets (Screen Size Dependent)
    pub targets: RenderTargets,

    // Shared Resources (Screen Size Independent)
    pub post_params_buffer: wgpu::Buffer,
    pub blit_params_buffer: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    // pub texture_array: wgpu::Texture, // Removed
    pub texture_view: wgpu::TextureView,

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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        // --- Texture Array ---
        // Use texture view from scene resources
        let texture_view = &scene_resources.texture_view;

        /*
        // --- Texture Array (0: White, 1: Checker) ---
        // MOVED TO SCENE BUILDER
         */

        // --- Passes ---
        let gbuffer_pass = GBufferPass::new(
            ctx,
            scene_resources,
            camera_buffer,
            &targets.gbuffer_pos_view,
            &targets.gbuffer_normal_view,
            &targets.gbuffer_albedo_view,
            &targets.gbuffer_motion_view,
            &texture_view,
            &sampler,
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

        let restir_spatial_pass = RestirSpatialPass::new(
            ctx,
            scene_resources,
            camera_buffer,
            &targets.gbuffer_pos_view,
            &targets.gbuffer_normal_view,
            &targets.gbuffer_albedo_view,
            &targets.gbuffer_motion_view,
            &restir_pass.reservoir_buffers[0], // Input (Temporal Result)
            &restir_pass.reservoir_buffers[1], // Output (Spatial Result)
            &targets.raw_view,                 // Output (Color)
            render_width,
            render_height,
        );

        let post_pass = PostPass::new(
            ctx,
            &targets.raw_view,
            &targets.accumulation_buffer,
            &targets.pp_view,
            &post_params_buffer,
            &targets.gbuffer_normal_view,
            &targets.gbuffer_pos_view,
            &targets.gbuffer_motion_view, // New
        );

        let blit_pass = BlitPass::new(ctx, &targets.pp_view, &sampler, &blit_params_buffer);

        Self {
            render_width,
            render_height,
            window_width: ctx.config.width,
            window_height: ctx.config.height,
            gbuffer_pass,
            restir_pass,
            restir_spatial_pass,
            // shade_pass: ShadePass::new(ctx), // Removed
            post_pass,
            blit_pass,
            targets,
            post_params_buffer,
            blit_params_buffer,
            sampler,
            // texture_array, // Removed
            texture_view: texture_view.clone(),
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
        _camera_buffer: &wgpu::Buffer,
        _light_buffer: &wgpu::Buffer,
        _material_buffer: &wgpu::Buffer,
        _attribute_buffer: &wgpu::Buffer,
        _index_buffer: &wgpu::Buffer,
        _mesh_info_buffer: &wgpu::Buffer,
        _tlas: &wgpu::Tlas,
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
        self.gbuffer_pass.execute(
            &mut encoder,
            self.render_width,
            self.render_height,
            self.frame_count,
        );

        // ---------------------------------------------------------------------
        // DEBUG: G-Buffer Visualization
        // 0: Shaded, 1: Pos (Float), 2: Normal (Float), 3: Albedo (Unorm), 4: Motion
        let debug_mode = 0;

        if debug_mode == 0 {
            // 1. ReSTIR Temporal Pass (Compute reservoirs)
            // Reads from Buffers[1] (Spatial/Prev Result), Writes to Buffers[0] (Temporal/Curr Result)
            self.restir_pass.execute(
                &mut encoder,
                ctx,
                self.render_width,
                self.render_height,
                self.frame_count,
                light_count,
                &self.texture_view,
                &self.sampler,
            );

            // 2. ReSTIR Spatial Pass (AND Shading)
            // Reads from Buffers[0] (Temporal Result), Writes to Buffers[1] (Spatial Final Result) + Output Texture
            self.restir_spatial_pass.execute(
                &mut encoder,
                ctx,
                self.render_width,
                self.render_height,
                self.frame_count,
                light_count,
                &self.texture_view,
                &self.sampler,
            );

            // 3. Post Pass (Tonemap + Accumulate?)
            // RestirSpatialPass writes to raw_view.
            // PostPass reads raw_view, writes to pp_view.
            self.post_pass.execute(
                &mut encoder,
                self.render_width,
                self.render_height,
                self.frame_count,
            );
        } else if debug_mode == 1 || debug_mode == 2 || debug_mode == 4 {
            // Visualize Float Texture (Pos / Normal / Motion) -> PostPass (Tonemap)
            let idx = (self.frame_count % 2) as usize;
            let src_tex = if debug_mode == 1 {
                &self.targets.gbuffer_pos[idx]
            } else if debug_mode == 2 {
                &self.targets.gbuffer_normal[idx]
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
            self.post_pass.execute(
                &mut encoder,
                self.render_width,
                self.render_height,
                self.frame_count,
            );
        } else {
            // Visualize Albedo (Unorm) -> Skip PostPass
            let idx = (self.frame_count % 2) as usize;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.targets.gbuffer_albedo[idx], // Fix: double buffer
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
