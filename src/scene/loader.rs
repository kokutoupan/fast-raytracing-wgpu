use crate::geometry::{self, Geometry, VertexAttributes};
use crate::scene::Material;
use anyhow::Result;
use gltf::mesh::util::ReadIndices;

use image::{DynamicImage, ImageBuffer, Rgba};

pub fn load_gltf(
    path: &str,
    device: &wgpu::Device,
) -> Result<(Vec<Geometry>, Vec<Material>, Vec<DynamicImage>)> {
    let (document, buffers, images) = gltf::import(path)?;

    let mut geometries = Vec::new();
    let mut materials = Vec::new();
    let mut loaded_images = Vec::new();

    // 0. Load Images
    for image in images {
        // Convert to DynamicImage (Rgba8)
        let img = match image.format {
            gltf::image::Format::R8G8B8 => {
                let buffer = image.pixels;
                DynamicImage::ImageRgb8(
                    ImageBuffer::from_raw(image.width, image.height, buffer).unwrap(),
                )
            }
            gltf::image::Format::R8G8B8A8 => {
                let buffer = image.pixels;
                DynamicImage::ImageRgba8(
                    ImageBuffer::from_raw(image.width, image.height, buffer).unwrap(),
                )
            }
            _ => {
                println!("Unsupported image format: {:?}", image.format);
                // Return dummy white image
                DynamicImage::ImageRgba8(ImageBuffer::from_fn(512, 512, |_, _| {
                    Rgba([255, 255, 255, 255])
                }))
            }
        };

        // Resize to 512x512 for now (to fit in our simple texture array)
        let resized = img.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
        loaded_images.push(resized);
    }

    // 1. Load Materials
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let metallic = pbr.metallic_factor();
        let roughness = pbr.roughness_factor();

        let mut mat = Material::new(base_color)
            .metallic(metallic)
            .roughness(roughness)
            .texture(u32::MAX); // Default to no-texture

        if let Some(texture_info) = pbr.base_color_texture() {
            // Need to offset index by existing textures (2: white + checker)
            // But here we just return raw index, mapping will happen in scene builder
            mat = mat.texture(texture_info.texture().source().index() as u32);
        }

        materials.push(mat);
    }

    // Default material if none exists
    if materials.is_empty() {
        materials.push(Material::new([1.0, 1.0, 1.0, 1.0]));
    }

    // 2. Load Meshes
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mut positions = Vec::new();
            let mut attributes = Vec::new();
            let mut indices = Vec::new();

            // Positions
            if let Some(iter) = reader.read_positions() {
                for pos in iter {
                    positions.push([pos[0], pos[1], pos[2], 1.0]);
                }
            }

            // Normals
            let normals: Vec<[f32; 3]> = if let Some(iter) = reader.read_normals() {
                iter.collect()
            } else {
                // Generate dummy normals if missing (should properly calculate)
                vec![[0.0, 1.0, 0.0]; positions.len()]
            };

            // UVs
            let uvs: Vec<[f32; 2]> = if let Some(iter) = reader.read_tex_coords(0) {
                iter.into_f32().collect()
            } else {
                vec![[0.0, 0.0]; positions.len()]
            };

            // Assemble VertexAttributes
            for (i, &n) in normals.iter().enumerate() {
                let encoded_normal = geometry::encode_octahedral_normal(n);
                attributes.push(VertexAttributes {
                    normal: encoded_normal,
                    uv: uvs.get(i).cloned().unwrap_or([0.0, 0.0]),
                });
            }

            // Indices
            if let Some(read_indices) = reader.read_indices() {
                match read_indices {
                    ReadIndices::U8(iter) => indices.extend(iter.map(|x| x as u32)),
                    ReadIndices::U16(iter) => indices.extend(iter.map(|x| x as u32)),
                    ReadIndices::U32(iter) => indices.extend(iter),
                }
            } else {
                // Non-indexed: generate sequential indices
                indices.extend(0..positions.len() as u32);
            }

            // Create Geometry (BLAS)
            let geo = geometry::build_blas(
                device,
                &format!("GLTF Mesh {}", mesh.index()),
                positions,
                attributes,
                indices,
            );

            geometries.push(geo);
        }
    }

    Ok((geometries, materials, loaded_images))
}
