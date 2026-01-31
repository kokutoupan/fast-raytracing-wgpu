use crate::geometry::{self, Geometry, VertexAttributes};
use crate::scene::Material;
use anyhow::Result;
use gltf::mesh::util::ReadIndices;

pub fn load_gltf(path: &str, device: &wgpu::Device) -> Result<(Vec<Geometry>, Vec<Material>)> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut geometries = Vec::new();
    let mut materials = Vec::new();

    // 1. Load Materials
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let metallic = pbr.metallic_factor();
        let roughness = pbr.roughness_factor();

        // IOR and Transmission extension handling could be added here
        // For now, use basic PBR
        let mut mat = Material::new(base_color)
            .metallic(metallic)
            .roughness(roughness);

        if let Some(texture_info) = pbr.base_color_texture() {
            mat = mat.texture(texture_info.texture().index() as u32);
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

    Ok((geometries, materials))
}
