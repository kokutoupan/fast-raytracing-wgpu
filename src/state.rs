use crate::camera::CameraController;
use crate::renderer::Renderer;
use crate::scene;
use crate::screenshot::ScreenshotTask;
use crate::wgpu_ctx::WgpuContext;
use crate::wgpu_utils::*;
use std::sync::Arc;

pub struct State {
    pub ctx: WgpuContext,
    pub renderer: Renderer,

    pub window: Arc<winit::window::Window>,

    // シーン資源
    pub scene_resources: scene::SceneResources,

    // カメラ関連
    pub camera_buffer: wgpu::Buffer,
    pub camera_controller: CameraController,

    // Screenshot & Control
    pub is_paused: bool,
    pub screenshot_requested: bool,

    // スクリーンショット用
    pub screenshot_buffer: wgpu::Buffer,
    pub screenshot_padded_bytes_per_row: u32,
    pub screenshot_sender: std::sync::mpsc::Sender<ScreenshotTask>,
    pub auto_screenshot_done: bool,
}

impl State {
    pub async fn new(window: winit::window::Window, render_size: (u32, u32)) -> Self {
        let window = Arc::new(window);
        let (screenshot_sender, screenshot_receiver) = std::sync::mpsc::channel::<ScreenshotTask>();

        // スクリーンショット保存用スレッド
        std::thread::spawn(move || {
            let mut saver = crate::screenshot::ScreenshotSaver::new();
            while let Ok(task) = screenshot_receiver.recv() {
                saver.process_and_save(task);
            }
        });

        // 1. WGPU初期化
        let ctx = WgpuContext::new(window.clone()).await;

        // 2. シーン構築
        let scene_resources = scene::create_restir_scene(&ctx.device, &ctx.queue);
        // let scene_resources = scene::create_cornell_box(&ctx.device, &ctx.queue);

        // 3. カメラ初期化 (最初はデフォルトのアスペクト比で初期化)
        let camera_controller = CameraController::new();
        let camera_uniform = camera_controller.build_uniform(1.0, 0, 0);

        let camera_buffer = create_buffer_init(
            &ctx.device,
            "Camera Buffer",
            &[camera_uniform],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // 4. レンダラー初期化
        let (render_width, render_height) = render_size;
        let renderer = Renderer::new(
            &ctx,
            &scene_resources,
            &camera_buffer,
            render_width,
            render_height,
        );

        // レンダラーのアスペクト比を使ってカメラユニフォームを更新
        let camera_uniform = camera_controller.build_uniform(renderer.aspect_ratio(), 0, 0);
        ctx.queue
            .write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // 5. スクリーンショット準備 (内部解像度で固定)
        let screenshot_padded_bytes_per_row = get_padded_bytes_per_row(render_width);
        let screenshot_buffer = create_buffer(
            &ctx.device,
            "Screenshot Buffer",
            (screenshot_padded_bytes_per_row * render_height) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        Self {
            ctx,
            renderer,
            window,
            scene_resources,
            camera_buffer,
            camera_controller,
            is_paused: false,
            screenshot_requested: false,
            screenshot_buffer,
            screenshot_padded_bytes_per_row,
            screenshot_sender,
            auto_screenshot_done: false,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.ctx.resize(new_size);

            self.renderer
                .resize(&self.ctx, &self.scene_resources, &self.camera_buffer);

            // スクリーンショットバッファは内部解像度固定なので再確保不要
        }
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) {
        if let winit::event::WindowEvent::KeyboardInput { event: ev, .. } = event {
            if ev.state == winit::event::ElementState::Pressed && !ev.repeat {
                match ev.logical_key {
                    winit::keyboard::Key::Character(ref s) if s == "j" => {
                        self.is_paused = !self.is_paused;
                    }
                    winit::keyboard::Key::Character(ref s) if s == "k" => {
                        self.screenshot_requested = true;
                    }
                    _ => (),
                }
            }
        }
        self.camera_controller.process_events(event);
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        if self.is_paused {
            return;
        }

        if self.camera_controller.update_camera(dt) {
            self.renderer.frame_count = 0;
        }

        let camera_uniform = self.camera_controller.build_uniform(
            self.renderer.aspect_ratio(),
            self.renderer.frame_count,
            self.scene_resources.num_lights,
        );
        self.ctx.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.is_paused {
            return Ok(());
        }

        let output = self.ctx.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render(
            &self.ctx,
            &view,
            &self.camera_buffer,
            &self.scene_resources.light_buffer,
            &self.scene_resources.material_buffer,
            &self.scene_resources.global_vertex_buffer,
            &self.scene_resources.global_index_buffer,
            &self.scene_resources.mesh_info_buffer,
            &self.scene_resources.tlas,
            self.scene_resources.num_lights,
        )?;

        // 自動スクリーンショット (検証用: 最初の1回だけ)
        const TARGET_SPP: u32 = 64;
        if !self.auto_screenshot_done && self.renderer.frame_count == TARGET_SPP {
            println!(
                "Target SPP ({}) reached! Taking one-time automatic screenshot...",
                TARGET_SPP
            );
            self.screenshot_requested = true;
            self.auto_screenshot_done = true;
        }

        if self.screenshot_requested {
            self.save_screenshot(&self.renderer.targets.post_processed_texture);
            self.screenshot_requested = false;
        }

        output.present();
        Ok(())
    }

    pub fn save_screenshot(&self, src_texture: &wgpu::Texture) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.screenshot_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.screenshot_padded_bytes_per_row),
                    rows_per_image: Some(self.renderer.render_height),
                },
            },
            wgpu::Extent3d {
                width: self.renderer.render_width,
                height: self.renderer.render_height,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // GPUの処理待ち & 読み取り
        let buffer_slice = self.screenshot_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Ok(Ok(_)) = rx.recv() {
            let data = buffer_slice.get_mapped_range().to_vec();
            self.screenshot_buffer.unmap();

            let task = ScreenshotTask {
                data,
                width: self.renderer.render_width,
                height: self.renderer.render_height,
                padded_bytes_per_row: self.screenshot_padded_bytes_per_row,
            };
            self.screenshot_sender.send(task).unwrap();
        }
    }
}
