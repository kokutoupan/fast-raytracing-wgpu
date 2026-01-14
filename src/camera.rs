use glam::{Mat4, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_inverse: [[f32; 4]; 4],
    pub proj_inverse: [[f32; 4]; 4],
    pub frame_count: u32,
    pub num_lights: u32,
    pub _padding: [u32; 2], // 16バイトアライメントのためのパディング
}

pub struct CameraController {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,

    // 入力状態
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,

    is_left_turn_pressed: bool,
    is_right_turn_pressed: bool,
    is_up_turn_pressed: bool,
    is_down_turn_pressed: bool,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0), // 初期位置: Z=3.0
            yaw: -90.0_f32.to_radians(),        // 初期向き: Zマイナス方向
            pitch: 0.0,

            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_left_turn_pressed: false,
            is_right_turn_pressed: false,
            is_up_turn_pressed: false,
            is_down_turn_pressed: false,
        }
    }

    pub fn process_events(&mut self, event: &winit::event::WindowEvent) -> bool {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let is_pressed = key_event.state.is_pressed();
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft) => {
                        self.is_down_pressed = is_pressed;
                        true
                    }

                    PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        self.is_left_turn_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                        self.is_right_turn_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        self.is_up_turn_pressed = is_pressed;
                        true
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        self.is_down_turn_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, dt: std::time::Duration) -> bool {
        let dt_secs = dt.as_secs_f32();
        let speed = 2.0 * dt_secs;
        let rotate_speed = 1.5 * dt_secs;

        let mut moved = false;

        // 回転の更新
        if self.is_right_turn_pressed {
            self.yaw += rotate_speed;
            moved = true;
        }
        if self.is_left_turn_pressed {
            self.yaw -= rotate_speed;
            moved = true;
        }
        if self.is_up_turn_pressed {
            self.pitch += rotate_speed;
            moved = true;
        }
        if self.is_down_turn_pressed {
            self.pitch -= rotate_speed;
            moved = true;
        }

        // クランプ (真上・真下を見過ぎないように)
        let old_pitch = self.pitch;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        if self.pitch != old_pitch {
            moved = true;
        }

        // 前方ベクトル・右ベクトルの計算
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();

        let forward = Vec3::new(cos_p * cos_y, sin_p, cos_p * sin_y).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = Vec3::Y;

        // 移動の更新
        if self.is_forward_pressed {
            self.position += forward * speed;
            moved = true;
        }
        if self.is_backward_pressed {
            self.position -= forward * speed;
            moved = true;
        }
        if self.is_right_pressed {
            self.position += right * speed;
            moved = true;
        }
        if self.is_left_pressed {
            self.position -= right * speed;
            moved = true;
        }
        if self.is_up_pressed {
            self.position += up * speed;
            moved = true;
        }
        if self.is_down_pressed {
            self.position -= up * speed;
            moved = true;
        }

        moved
    }

    pub fn build_uniform(&self, aspect: f32, frame_count: u32, num_lights: u32) -> CameraUniform {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let forward = Vec3::new(cos_p * cos_y, sin_p, cos_p * sin_y).normalize();

        let view = Mat4::look_at_rh(self.position, self.position + forward, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);

        CameraUniform {
            view_inverse: view.inverse().to_cols_array_2d(),
            proj_inverse: proj.inverse().to_cols_array_2d(),
            frame_count,
            num_lights,
            _padding: [0; 2],
        }
    }
}
