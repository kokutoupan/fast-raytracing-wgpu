use crate::camera::CameraController;
use crate::scene;
use image::ImageEncoder; // トレイトのインポート
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct ScreenshotTask {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    data: Vec<u8>,
}

pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<winit::window::Window>,

    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,

    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group: wgpu::BindGroup,

    // Resize用のレイアウト保持
    pub compute_bind_group_layout: wgpu::BindGroupLayout,
    pub blit_bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,

    // シーン資源
    pub scene_resources: scene::SceneResources,

    // カメラ関連
    pub camera_buffer: wgpu::Buffer,
    pub camera_controller: CameraController,

    // アキュムレーション関連
    pub accumulation_buffer: wgpu::Buffer,
    pub frame_count: u32,

    // Screenshot & Control
    pub storage_texture: wgpu::Texture,
    pub is_paused: bool,
    pub screenshot_requested: bool,

    // スクリーンショット用
    pub screenshot_buffer: wgpu::Buffer,
    pub screenshot_padded_bytes_per_row: u32,
    pub screenshot_sender: std::sync::mpsc::Sender<ScreenshotTask>,
}

impl State {
    pub async fn new(window: winit::window::Window) -> Self {
        let window = Arc::new(window);
        let (screenshot_sender, screenshot_receiver) = std::sync::mpsc::channel::<ScreenshotTask>();

        // スクリーンショット保存用スレッド
        std::thread::spawn(move || {
            while let Ok(task) = screenshot_receiver.recv() {
                save_image(task);
            }
        });

        // 1. 初期化
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::from_env_or_default(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .unwrap();
        let mut config = surface
            .get_default_config(
                &adapter,
                window.inner_size().width,
                window.inner_size().height,
            )
            .unwrap();
        config.usage |= wgpu::TextureUsages::COPY_SRC;
        surface.configure(&device, &config);

        // 2. シーン構築
        let scene_resources = scene::create_cornell_box(&device, &queue);

        // 3. カメラ初期化
        let camera_controller = CameraController::new();
        let camera_uniform =
            camera_controller.build_uniform(config.width as f32 / config.height as f32, 0);

        let camera_buffer = create_buffer_init(
            &device,
            "Camera Buffer",
            &[camera_uniform],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // 4. パイプライン
        let accumulation_buffer = create_buffer(
            &device,
            "Accumulation Buffer",
            (config.width * config.height * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let storage_tex = create_storage_texture(&device, &config);
        let storage_view = storage_tex.create_view(&Default::default());

        // --- 追加: スクリーンショット用バッファの事前確保 ---
        let screenshot_padded_bytes_per_row = get_padded_bytes_per_row(config.width);
        let screenshot_buffer = create_buffer(
            &device,
            "Screenshot Buffer",
            (screenshot_padded_bytes_per_row * config.height) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // --- 定数の設定 ---
        let constants = [("MAX_DEPTH", 8.0), ("SPP", 2.0)];

        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("blit.wgsl"));

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Global Vertex Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Global Index Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Mesh Info Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&compute_bgl],
                    immediate_size: 0,
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });

        let compute_bind_group = create_compute_bind_group(
            &device,
            &compute_bgl,
            &scene_resources,
            &storage_view,
            &camera_buffer,
            &accumulation_buffer,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&blit_bgl],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let blit_bind_group = create_blit_bind_group(&device, &blit_bgl, &storage_view, &sampler);

        Self {
            device,
            queue,
            surface,
            config,
            window,
            compute_pipeline,
            compute_bind_group,
            blit_pipeline,
            blit_bind_group,
            camera_buffer,
            camera_controller,
            accumulation_buffer,
            frame_count: 0,
            compute_bind_group_layout: compute_bgl,
            blit_bind_group_layout: blit_bgl,
            sampler,
            scene_resources,
            storage_texture: storage_tex,
            is_paused: false,
            screenshot_requested: false,
            screenshot_buffer,
            screenshot_padded_bytes_per_row,
            screenshot_sender,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.accumulation_buffer = create_buffer(
                &self.device,
                "Accumulation Buffer",
                (self.config.width * self.config.height * 16) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );

            let storage_tex = create_storage_texture(&self.device, &self.config);
            let storage_view = storage_tex.create_view(&Default::default());

            self.storage_texture = storage_tex;

            // --- 追加: スクリーンショット用バッファの再確保 ---
            self.screenshot_padded_bytes_per_row = get_padded_bytes_per_row(self.config.width);
            self.screenshot_buffer = create_buffer(
                &self.device,
                "Screenshot Buffer",
                (self.screenshot_padded_bytes_per_row * self.config.height) as u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );

            self.compute_bind_group = create_compute_bind_group(
                &self.device,
                &self.compute_bind_group_layout,
                &self.scene_resources,
                &storage_view,
                &self.camera_buffer,
                &self.accumulation_buffer,
            );

            self.blit_bind_group = create_blit_bind_group(
                &self.device,
                &self.blit_bind_group_layout,
                &storage_view,
                &self.sampler,
            );

            self.frame_count = 0;
        }
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) {
        self.camera_controller.process_events(event);
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        state: winit::event::ElementState::Pressed,
                        physical_key: winit::keyboard::PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => match keycode {
                winit::keyboard::KeyCode::KeyJ => {
                    self.is_paused = !self.is_paused;
                    println!("Paused: {}", self.is_paused);
                }
                winit::keyboard::KeyCode::KeyK => {
                    self.screenshot_requested = true;
                    println!("Screenshot requested");
                }
                _ => {}
            },
            _ => {}
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        let prev_pos = self.camera_controller.position;
        let prev_yaw = self.camera_controller.yaw;
        let prev_pitch = self.camera_controller.pitch;

        self.camera_controller.update_camera(dt);

        if self.camera_controller.position != prev_pos
            || self.camera_controller.yaw != prev_yaw
            || self.camera_controller.pitch != prev_pitch
        {
            self.frame_count = 0;
        } else {
            self.frame_count += 1;
        }

        let uniform = self.camera_controller.build_uniform(
            self.config.width as f32 / self.config.height as f32,
            self.frame_count,
        );
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if !self.is_paused {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.compute_bind_group), &[]);
            cpass.dispatch_workgroups(
                self.config.width.div_ceil(8),
                self.config.height.div_ceil(8),
                1,
            );
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, Some(&self.blit_bind_group), &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Check for screenshot request
        if self.screenshot_requested {
            self.save_screenshot(&output.texture);
            self.screenshot_requested = false;
        }

        output.present();
        Ok(())
    }

    // src/state.rs の State impl内

    fn save_screenshot(&self, src_texture: &wgpu::Texture) {
        let saving_start = chrono::Local::now(); // 計測開始

        let width = self.config.width;
        let height = self.config.height;
        let padded_bytes_per_row = self.screenshot_padded_bytes_per_row;

        // 1. コピーコマンドの発行
        let mut encoder = self.device.create_command_encoder(&Default::default());
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
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // 2. マップと待機
        let buffer_slice = self.screenshot_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // ここでGPU完了待ち（これが一番長いが避けられない）
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        if receiver.recv().unwrap().is_ok() {
            // 3. データの吸い出し (ここを最速にする)
            let data = buffer_slice.get_mapped_range();

            let raw_data = data.to_vec();

            drop(data);
            self.screenshot_buffer.unmap();

            // メインスレッドの仕事はここまで！
            // 計測終了
            println!(
                "Main thread freeze time: {}ms",
                chrono::Local::now().timestamp_millis() - saving_start.timestamp_millis()
            );

            let width = self.config.width;
            let height = self.config.height;
            let padded_bytes_per_row = self.screenshot_padded_bytes_per_row;

            // スレッド処理開始 (Channel送信に変更)
            let _ = self.screenshot_sender.send(ScreenshotTask {
                width,
                height,
                padded_bytes_per_row,
                data: raw_data,
            });
        }
    }
}

fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn create_buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn create_storage_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Storage Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn get_padded_bytes_per_row(width: u32) -> u32 {
    let unpadded_bytes_per_row = width * 4;
    let align = 256;
    let padding = (align - unpadded_bytes_per_row % align) % align;
    unpadded_bytes_per_row + padding
}

fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_resources: &scene::SceneResources,
    storage_view: &wgpu::TextureView,
    camera_buffer: &wgpu::Buffer,
    accumulation_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&scene_resources.tlas),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scene_resources.material_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: accumulation_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scene_resources.global_vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scene_resources.global_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scene_resources.mesh_info_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_blit_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    storage_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn save_image(task: ScreenshotTask) {
    let ScreenshotTask {
        width,
        height,
        padded_bytes_per_row,
        data: raw_data,
    } = task;

    let saving_start = chrono::Local::now();
    let unpadded_bytes_per_row = (width * 4) as usize;
    let padded_bytes_per_row = padded_bytes_per_row as usize;

    // ========================================================================
    // 分岐: AIデノイズ (feature = "ai-denoise") vs 爆速モード (それ以外)
    // ========================================================================

    // 最終的に保存する画像バッファを受け取る変数
    let final_image_buffer: image::RgbaImage;

    #[cfg(feature = "ai-denoise")]
    {
        // -----------------------------------------------------------
        // 【AIモード】: 高画質・低速 (1.0s~)
        // -----------------------------------------------------------
        println!("Process Mode: AI Denoise (High Quality)");

        // 1. パディング除去 (u8)
        let mut image_data = Vec::with_capacity((width * height * 4) as usize);
        for chunk in raw_data.chunks(padded_bytes_per_row) {
            image_data.extend_from_slice(&chunk[..unpadded_bytes_per_row]);
        }

        // 2. ImageBuffer作成
        let mut image_buffer =
            image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, image_data)
                .expect("Dimension mismatch");

        let device = oidn::Device::new();
        let mut filter = oidn::RayTracing::new(&device);
        filter.srgb(true);
        filter.image_dimensions(width as usize, height as usize);

        // 3. BGRA(u8) -> RGB(f32) 変換 (Rayon並列化)
        let mut input_rgb = vec![0.0f32; (width * height * 3) as usize];
        image_buffer
            .as_raw()
            .par_chunks(4)
            .zip(input_rgb.par_chunks_mut(3))
            .for_each(|(bgra, rgb)| {
                rgb[0] = bgra[2] as f32 / 255.0; // R
                rgb[1] = bgra[1] as f32 / 255.0; // G
                rgb[2] = bgra[0] as f32 / 255.0; // B
            });

        let mut output_rgb = vec![0.0f32; input_rgb.len()];

        // 4. AI推論実行 (激重ポイント)
        filter.filter(&input_rgb, &mut output_rgb).unwrap();
        if let Err(e) = device.get_error() {
            eprintln!("OIDN Error: {}", e.1);
        }

        // 5. RGB(f32) -> RGBA(u8) 書き戻し (Rayon並列化)
        image_buffer
            .as_mut()
            .par_chunks_mut(4)
            .zip(output_rgb.par_chunks(3))
            .for_each(|(rgba, rgb)| {
                rgba[0] = (rgb[0] * 255.0).clamp(0.0, 255.0) as u8;
                rgba[1] = (rgb[1] * 255.0).clamp(0.0, 255.0) as u8;
                rgba[2] = (rgb[2] * 255.0).clamp(0.0, 255.0) as u8;
                rgba[3] = 255;
            });

        final_image_buffer = image_buffer;
    }

    #[cfg(not(feature = "ai-denoise"))]
    {
        // -----------------------------------------------------------
        // 【爆速モード】: 低画質・高速 (50ms~100ms)
        // -----------------------------------------------------------
        println!("Process Mode: Fast Blur (High Speed)");

        // 1. パディング除去 & BGRA->RGBA変換 & u8生成 (一撃で行う)
        let mut rgba_data = vec![0u8; (width * height * 4) as usize];

        rgba_data
            .par_chunks_mut(unpadded_bytes_per_row)
            .zip(raw_data.par_chunks(padded_bytes_per_row))
            .for_each(|(dest_row, src_row)| {
                for (dest_pixel, src_pixel) in
                    dest_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4))
                {
                    dest_pixel[0] = src_pixel[2]; // R (Src:B)
                    dest_pixel[1] = src_pixel[1]; // G (Src:G)
                    dest_pixel[2] = src_pixel[0]; // B (Src:R)
                    dest_pixel[3] = 255; // A
                }
            });

        let temp_buffer =
            image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba_data)
                .expect("Dimension mismatch");

        final_image_buffer = temp_buffer;
    }

    // ========================================================================
    // 共通: 保存処理 (PNG高速設定)
    // ========================================================================
    let now = chrono::Local::now();
    let filename = format!("output/screenshot_{}.png", now.format("%Y-%m-%d_%H-%M-%S"));
    let _ = std::fs::create_dir_all("output");

    let file = File::create(&filename).unwrap();
    let ref mut w = BufWriter::new(file);

    let encoder = PngEncoder::new_with_quality(
        w,
        CompressionType::Fast, // 爆速設定
        FilterType::NoFilter,
    );

    encoder
        .write_image(
            &final_image_buffer,
            width,
            height,
            image::ColorType::Rgba8.into(),
        )
        .unwrap();

    println!(
        "Saved screenshot: {} ({}ms)",
        filename,
        chrono::Local::now().timestamp_millis() - saving_start.timestamp_millis()
    );
}
