mod camera;
mod geometry;
mod passes;
mod renderer;
mod scene;
mod screenshot;
mod state;
mod wgpu_ctx;
mod wgpu_utils;
use state::State;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};

struct App {
    state: Option<State>,
    last_fps_update: Option<Instant>,
    frame_count: u32,
    last_frame_time: Option<Instant>,
    render_size: (u32, u32),
}
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        let window = el
            .create_window(
                winit::window::Window::default_attributes().with_title("RayQuery Camera"),
            )
            .unwrap();
        self.state = Some(pollster::block_on(State::new(window, self.render_size)));
        self.last_fps_update = Some(Instant::now());
        self.frame_count = 0;
        self.last_frame_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        el: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            // 1. 入力を処理
            state.input(&ev);

            match ev {
                WindowEvent::CloseRequested => el.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                    state.window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    let physical_size = state.window.inner_size();
                    state.resize(physical_size);
                    state.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    // 2. カメラ位置更新 & バッファ転送
                    let now = Instant::now();
                    let dt = now - self.last_frame_time.unwrap_or(now);
                    self.last_frame_time = Some(now);

                    state.update(dt);

                    // 3. 描画
                    match state.render() {
                        Ok(_) => {}
                        // サーフェスが古いかロストした場合はリサイズして再開
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            state.resize(state.window.inner_size());
                        }
                        // メモリ不足は終了
                        Err(wgpu::SurfaceError::OutOfMemory) => el.exit(),
                        // その他(Timeout)はログに流して次フレームへ
                        Err(e) => eprintln!("{:?}", e),
                    }

                    // FPS計算
                    self.frame_count += 1;
                    if let Some(last_update) = self.last_fps_update {
                        if last_update.elapsed().as_secs_f32() >= 1.0 {
                            let fps = self.frame_count as f32 / last_update.elapsed().as_secs_f32();
                            let width = state.ctx.config.width;
                            let height = state.ctx.config.height;
                            let samples = state.renderer.frame_count; // Frame count in renderer (accumulation count)
                            state.window.set_title(&format!(
                                "RayQuery Camera - FPS: {:.1} - Res: {}x{} - Samples: {}",
                                fps, width, height, samples
                            ));
                            self.frame_count = 0;
                            self.last_fps_update = Some(Instant::now());
                        }
                    }

                    state.window.request_redraw(); // 連続描画
                }
                _ => (),
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let render_size = args
        .iter()
        .find(|a| a.starts_with("--scale="))
        .and_then(|a| a.split('=').nth(1))
        .and_then(|s| {
            let parts: Vec<&str> = s.split('x').collect();
            if parts.len() == 2 {
                let w = parts[0].parse::<u32>().ok();
                let h = parts[1].parse::<u32>().ok();
                if let (Some(w), Some(h)) = (w, h) {
                    return Some((w, h));
                }
            }
            None
        })
        .unwrap_or((1280, 720));

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App {
            state: None,
            last_fps_update: None,
            frame_count: 0,
            last_frame_time: None,
            render_size,
        })
        .unwrap();
}
