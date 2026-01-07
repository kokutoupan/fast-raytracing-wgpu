mod camera;
mod state; // 追加
use state::State;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};

struct App {
    state: Option<State>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        let window = el
            .create_window(
                winit::window::Window::default_attributes().with_title("RayQuery Camera"),
            )
            .unwrap();
        self.state = Some(pollster::block_on(State::new(window)));
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
                WindowEvent::RedrawRequested => {
                    // 2. カメラ位置更新 & バッファ転送
                    state.update();

                    // 3. 描画
                    let _ = state.render();
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
    event_loop.run_app(&mut App { state: None }).unwrap();
}
