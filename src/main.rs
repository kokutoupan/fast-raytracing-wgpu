use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes};

struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        println!("--- resumed() が呼ばれました ---"); // これが出るか確認

        let window_attributes = WindowAttributes::default()
            .with_title("Rust Raytracer")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

        match event_loop.create_window(window_attributes) {
            Ok(window) => {
                println!("ウィンドウ作成成功: {:?}", window.id());
                self.window = Some(window);
            }
            Err(e) => println!("ウィンドウ作成失敗: {:?}", e),
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // イベントが来ているか確認
        // println!("Event: {:?}", event);

        match event {
            WindowEvent::CloseRequested => {
                println!("終了リクエスト");
                _event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    // 本来はここでwgpuのレンダリングを行う
                    window.pre_present_notify();
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { window: None };
    event_loop.run_app(&mut app).unwrap();
}
