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

pub fn save_image(task: ScreenshotTask) {
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
