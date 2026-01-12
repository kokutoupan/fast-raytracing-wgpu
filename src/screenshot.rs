use image::ImageEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

pub struct ScreenshotTask {
    pub width: u32,
    pub height: u32,
    pub padded_bytes_per_row: u32,
    pub data: Vec<u8>,
}

pub struct ScreenshotSaver {
    // Buffers for reuse
    image_data: Vec<u8>,

    #[cfg(feature = "ai-denoise")]
    input_rgb: Vec<f32>,

    #[cfg(feature = "ai-denoise")]
    output_rgb: Vec<f32>,

    #[cfg(feature = "ai-denoise")]
    device: oidn::Device,
}

impl ScreenshotSaver {
    pub fn new() -> Self {
        #[cfg(feature = "ai-denoise")]
        let device = oidn::Device::new();

        Self {
            image_data: Vec::new(),
            #[cfg(feature = "ai-denoise")]
            input_rgb: Vec::new(),
            #[cfg(feature = "ai-denoise")]
            output_rgb: Vec::new(),
            #[cfg(feature = "ai-denoise")]
            device,
        }
    }

    pub fn process_and_save(&mut self, task: ScreenshotTask) {
        let ScreenshotTask {
            width,
            height,
            padded_bytes_per_row,
            data: raw_data,
        } = task;

        let saving_start = chrono::Local::now();
        let unpadded_bytes_per_row = (width * 4) as usize;
        let padded_bytes_per_row = padded_bytes_per_row as usize;
        let pixel_count = (width * height) as usize;

        // 1. パディング除去 (常にimage_dataにRGBAでコピーされる)
        if self.image_data.len() != pixel_count * 4 {
            self.image_data.resize(pixel_count * 4, 0);
        }

        self.image_data
            .par_chunks_mut(unpadded_bytes_per_row)
            .zip(raw_data.par_chunks(padded_bytes_per_row))
            .for_each(|(dest_row, src_row)| {
                dest_row.copy_from_slice(&src_row[..unpadded_bytes_per_row]);
            });

        #[cfg(feature = "ai-denoise")]
        {
            // -----------------------------------------------------------
            // 【AIモード】: 高画質・低速 (1.0s~)
            // -----------------------------------------------------------
            println!("Process Mode: AI Denoise (High Quality)");

            let mut filter = oidn::RayTracing::new(&self.device);
            filter.srgb(true);
            filter.image_dimensions(width as usize, height as usize);

            // 2. RGBA(u8) -> RGB(f32) 変換
            if self.input_rgb.len() != pixel_count * 3 {
                self.input_rgb.resize(pixel_count * 3, 0.0);
            }

            self.image_data
                .par_chunks(4)
                .zip(self.input_rgb.par_chunks_mut(3))
                .for_each(|(rgba, rgb)| {
                    rgb[0] = rgba[0] as f32 / 255.0; // R
                    rgb[1] = rgba[1] as f32 / 255.0; // G
                    rgb[2] = rgba[2] as f32 / 255.0; // B
                });

            // バッファ再利用
            if self.output_rgb.len() != pixel_count * 3 {
                self.output_rgb.resize(pixel_count * 3, 0.0);
            }

            // 3. AI推論実行
            filter
                .filter(&self.input_rgb, &mut self.output_rgb)
                .unwrap();

            if let Err(e) = self.device.get_error() {
                eprintln!("OIDN Error: {}", e.1);
            }

            // 4. RGB(f32) -> RGBA(u8) 書き戻し
            self.output_rgb
                .par_chunks(3)
                .zip(self.image_data.par_chunks_mut(4))
                .for_each(|(rgb, rgba)| {
                    rgba[0] = (rgb[0] * 255.0).clamp(0.0, 255.0) as u8;
                    rgba[1] = (rgb[1] * 255.0).clamp(0.0, 255.0) as u8;
                    rgba[2] = (rgb[2] * 255.0).clamp(0.0, 255.0) as u8;
                    rgba[3] = 255;
                });
        }

        #[cfg(not(feature = "ai-denoise"))]
        {
            // -----------------------------------------------------------
            // 【爆速モード】: 高速 (50ms~100ms)
            // -----------------------------------------------------------
            println!("Process Mode: Fast (Native RGBA)");
            // パディング除去済みなのでそのままPNG保存へ
        }

        // ========================================================================
        // 共通: 保存処理
        // ========================================================================
        let now = chrono::Local::now();
        let filename = format!("output/screenshot_{}.png", now.format("%Y-%m-%d_%H-%M-%S"));
        let _ = std::fs::create_dir_all("output");

        let file = File::create(&filename).unwrap();
        let mut w = BufWriter::new(file);

        let encoder =
            PngEncoder::new_with_quality(&mut w, CompressionType::Fast, FilterType::NoFilter);

        encoder
            .write_image(
                &self.image_data,
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
}
