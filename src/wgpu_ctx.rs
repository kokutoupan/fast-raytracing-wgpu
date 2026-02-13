use std::sync::Arc;

pub struct WgpuContext {
    pub _instance: wgpu::Instance,
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuContext {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
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

        // アダプターの情報を取得
        let info = adapter.get_info();
        println!("Adapter: {} ({:?})", info.name, info.backend);
        println!("Driver: {}", info.driver_info);

        // 機能チェック
        let features = adapter.features();
        if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            println!("✅ Hardware Ray Tracing (Ray Query) is supported!");
        } else {
            println!("❌ Hardware Ray Tracing is NOT supported on this adapter.");
            panic!("This application requires hardware ray tracing support.");
        }

        if features.contains(wgpu::Features::FLOAT32_FILTERABLE) {
            println!("✅ Float32 Filtering is supported!");
        } else {
            println!("⚠️ Float32 Filtering is NOT supported. Fallback might be needed (or crash).");
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .unwrap();

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        config.format = config.format.remove_srgb_suffix();
        config.usage |= wgpu::TextureUsages::COPY_SRC;
        surface.configure(&device, &config);

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            surface,
            config,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
