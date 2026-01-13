use crate::passes::*;
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
pub struct Renderer {
    pub render_width: u32,
    pub render_height: u32,
    pub window_width: u32,
    pub window_height: u32,

    // Passes
    raytrace_pass: RaytracePass,
    post_pass: PostPass,
    blit_pass: BlitPass,

    // Shared Resources
    pub raw_raytrace_texture: wgpu::Texture,
    pub post_processed_texture: wgpu::Texture,
    pub accumulation_buffer: wgpu::Buffer,
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
        // --- Shared Resources ---
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pp_view = post_processed_texture.create_view(&Default::default());

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

        let texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });

        // --- Passes ---
        let raytrace_pass = RaytracePass::new(
            ctx,
            scene_resources,
            camera_buffer,
            &raw_view,
            &texture_view,
            &sampler,
        );

        let post_pass = PostPass::new(
            ctx,
            &raw_view,
            &accumulation_buffer,
            &pp_view,
            &post_params_buffer,
        );

        let blit_pass = BlitPass::new(ctx, &pp_view, &sampler, &blit_params_buffer);

        Self {
            render_width,
            render_height,
            window_width: ctx.config.width,
            window_height: ctx.config.height,
            raytrace_pass,
            post_pass,
            blit_pass,
            raw_raytrace_texture,
            post_processed_texture,
            accumulation_buffer,
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

        // 1. Ray Tracing Pass
        self.raytrace_pass
            .execute(&mut encoder, self.render_width, self.render_height);

        // 2. Post Pass
        self.post_pass
            .execute(&mut encoder, self.render_width, self.render_height);

        // 3. Blit Pass
        self.blit_pass.execute(&mut encoder, view);

        ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;

        Ok(())
    }
}
