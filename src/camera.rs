use glam::{Mat4, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_inverse: [[f32; 4]; 4],
    pub proj_inverse: [[f32; 4]; 4],
    pub view_pos: [f32; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub frame_count: u32,
    pub num_lights: u32,
    pub _padding: [u32; 2],
}

pub struct CameraController {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub prev_view_proj: Mat4,

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
            prev_view_proj: Mat4::IDENTITY,

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

    pub fn get_halton_jitter(index: u32, width: u32, height: u32) -> (f32, f32) {
        fn halton(mut i: u32, base: u32) -> f32 {
            let mut f = 1.0;
            let mut r = 0.0;
            while i > 0 {
                f /= base as f32;
                r += f * (i % base) as f32;
                i /= base;
            }
            r
        }

        // 8-phase Halton sequence is usually enough for TAA
        let idx = index;
        let halton_x = halton(idx + 1, 2) - 0.5;
        let halton_y = halton(idx + 1, 3) - 0.5;

        // Scale by pixel size to get NDC offset
        // NDC is [-1, 1], so pixel size is 2.0 / size
        (
            (halton_x * 0.) / width as f32,
            (halton_y * 0.) / height as f32,
        )
    }

    pub fn build_uniform(
        &self,
        aspect: f32,
        frame_count: u32,
        num_lights: u32,
        jitter: (f32, f32),
    ) -> (CameraUniform, [[f32; 4]; 4]) {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let forward = Vec3::new(cos_p * cos_y, sin_p, cos_p * sin_y).normalize();

        let view = Mat4::look_at_rh(self.position, self.position + forward, Vec3::Y);
        let proj_base = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);

        // Calculate Unjittered ViewProj
        let view_proj_unjittered = proj_base * view;

        // Apply Jitter to Projection Matrix (Shear)
        let mut proj_cols = proj_base.to_cols_array_2d();
        proj_cols[2][0] += jitter.0;
        proj_cols[2][1] += jitter.1;
        let proj = Mat4::from_cols_array_2d(&proj_cols);

        // 現在のViewProjection行列 (Jittered)
        let view_proj = proj * view;

        // 初回のみ Identity なので防ぐ
        let prev_view_proj = if self.prev_view_proj == Mat4::IDENTITY {
            view_proj_unjittered
        } else {
            self.prev_view_proj
        };

        // We send `prev_view_proj` (Unjittered N-1) to shader.
        // We use `view_proj` (Jittered N) for current frame rendering.

        (
            CameraUniform {
                view_proj: view_proj.to_cols_array_2d(),
                view_inverse: view.inverse().to_cols_array_2d(),
                proj_inverse: proj.inverse().to_cols_array_2d(),
                view_pos: [self.position.x, self.position.y, self.position.z, 1.0],
                prev_view_proj: prev_view_proj.to_cols_array_2d(),
                frame_count,
                num_lights,
                _padding: [0; 2],
            },
            view_proj_unjittered.to_cols_array_2d(),
        )
    }
}
