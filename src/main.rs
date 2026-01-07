mod camera;
mod state; // 追加
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
}
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        let window = el
            .create_window(
                winit::window::Window::default_attributes().with_title("RayQuery Camera"),
            )
            .unwrap();
        self.state = Some(pollster::block_on(State::new(window)));
        self.last_fps_update = Some(Instant::now());
        self.frame_count = 0;
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
                }
                WindowEvent::RedrawRequested => {
                    // 2. カメラ位置更新 & バッファ転送
                    state.update();

                    // 3. 描画
                    let _ = state.render();

                    // FPS計算
                    self.frame_count += 1;
                    if let Some(last_update) = self.last_fps_update {
                        if last_update.elapsed().as_secs_f32() >= 1.0 {
                            let fps = self.frame_count as f32 / last_update.elapsed().as_secs_f32();
                            let width = state.config.width;
                            let height = state.config.height;
                            state.window.set_title(&format!(
                                "RayQuery Camera - FPS: {:.1} - Res: {}x{}",
                                fps, width, height
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App {
            state: None,
            last_fps_update: None,
            frame_count: 0,
        })
        .unwrap();
}
