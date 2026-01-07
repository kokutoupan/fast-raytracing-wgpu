mod state;
use state::State;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

struct App {
    state: Option<State>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        let window = el.create_window(Window::default_attributes()).unwrap();
        self.state = Some(pollster::block_on(State::new(window)));
    }
    fn window_event(
        &mut self,
        el: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(s) = &mut self.state {
                    let _ = s.render();
                    s.window.request_redraw();
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App { state: None }).unwrap();
}
