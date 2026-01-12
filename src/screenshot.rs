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

        #[cfg(feature = "ai-denoise")]
        {
            // -----------------------------------------------------------
            // 【AIモード】: 高画質・低速 (1.0s~)
            // -----------------------------------------------------------
            println!("Process Mode: AI Denoise (High Quality)");

            // 1. パディング除去 (u8)
            self.image_data.clear();
            self.image_data.reserve(pixel_count * 4);

            for chunk in raw_data.chunks(padded_bytes_per_row) {
                self.image_data
                    .extend_from_slice(&chunk[..unpadded_bytes_per_row]);
            }

            let mut filter = oidn::RayTracing::new(&self.device);
            filter.srgb(true);
            filter.image_dimensions(width as usize, height as usize);

            // 3. BGRA(u8) -> RGB(f32) 変換 (Rayon並列化)
            if self.input_rgb.len() != pixel_count * 3 {
                self.input_rgb.resize(pixel_count * 3, 0.0);
            }

            self.image_data
                .par_chunks(4)
                .zip(self.input_rgb.par_chunks_mut(3))
                .for_each(|(bgra, rgb)| {
                    rgb[0] = bgra[2] as f32 / 255.0; // R
                    rgb[1] = bgra[1] as f32 / 255.0; // G
                    rgb[2] = bgra[0] as f32 / 255.0; // B
                });

            // バッファ再利用
            if self.output_rgb.len() != pixel_count * 3 {
                self.output_rgb.resize(pixel_count * 3, 0.0);
            }

            // 4. AI推論実行 (激重ポイント)
            filter
                .filter(&self.input_rgb, &mut self.output_rgb)
                .unwrap();

            if let Err(e) = self.device.get_error() {
                eprintln!("OIDN Error: {}", e.1);
            }

            // 5. RGB(f32) -> RGBA(u8) 書き戻し (Rayon並列化)
            // self.image_data に直接書き戻す
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
            // 【爆速モード】: 低画質・高速 (50ms~100ms)
            // -----------------------------------------------------------
            println!("Process Mode: Fast Blur (High Speed)");

            // 1. パディング除去 & BGRA->RGBA変換 & u8生成 (一撃で行う)
            // self.image_dataを再利用 (rgba_data は削除して一本化)
            if self.image_data.len() != pixel_count * 4 {
                self.image_data.resize(pixel_count * 4, 0);
            }

            self.image_data
                .par_chunks_mut(unpadded_bytes_per_row)
                .zip(raw_data.par_chunks(padded_bytes_per_row))
                .for_each(|(dest_row, src_row)| {
                    for (dest_pixel, src_pixel) in
                        dest_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4))
                    {
                        dest_pixel[0] = src_pixel[0]; // R (Src:B)
                        dest_pixel[1] = src_pixel[1]; // G (Src:G)
                        dest_pixel[2] = src_pixel[2]; // B (Src:R)
                        dest_pixel[3] = 255; // A
                    }
                });
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
                &self.image_data, // 直接スライスを渡す
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
