use std::collections::HashMap;

// 頂点属性データ (32バイト) - Shaderに送る用
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexAttributes {
    pub normal: [f32; 4], // vec4f alignment
    pub uv: [f32; 4],     // uv + padding
}

pub struct Geometry {
    pub blas: wgpu::Blas,
    pub positions: Vec<[f32; 4]>,
    pub attributes: Vec<VertexAttributes>,
    pub indices: Vec<u32>,
    pub desc: wgpu::BlasTriangleGeometrySizeDescriptor,
}

fn build_blas(
    device: &wgpu::Device,
    label: &str,
    positions: Vec<[f32; 4]>,
    attributes: Vec<VertexAttributes>, // Keep strictly for Geometry struct
    indices: Vec<u32>,
) -> Geometry {
    let desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: positions.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some(label),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![desc.clone()],
        },
    );

    Geometry {
        blas,
        positions,
        attributes,
        indices,
        desc,
    }
}

// --- ヘルパー関数: 平面(Quad)のBLASを作成 ---
pub fn create_plane_blas(device: &wgpu::Device) -> Geometry {
    // 1x1 の平面 (XZ平面, 中心0,0)
    let positions = vec![
        [-0.5, 0.0, 0.5, 1.0],  // 左手前
        [0.5, 0.0, 0.5, 1.0],   // 右手前
        [-0.5, 0.0, -0.5, 1.0], // 左奥
        [0.5, 0.0, -0.5, 1.0],  // 右奥
    ];
    let attributes = vec![
        VertexAttributes {
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [0.0, 1.0, 0.0, 0.0],
        },
        VertexAttributes {
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [1.0, 1.0, 0.0, 0.0],
        },
        VertexAttributes {
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [0.0, 0.0, 0.0, 0.0],
        },
        VertexAttributes {
            normal: [0.0, 1.0, 0.0, 0.0],
            uv: [1.0, 0.0, 0.0, 0.0],
        },
    ];
    let indices: Vec<u32> = vec![0, 1, 2, 2, 1, 3]; // Triangle List

    build_blas(device, "Quad BLAS", positions, attributes, indices)
}

// --- ヘルパー関数: 立方体(Cube)のBLASを作成 ---
pub fn create_cube_blas(device: &wgpu::Device) -> Geometry {
    let mut positions = Vec::new();
    let mut attributes = Vec::new();
    let mut indices = Vec::new();
    let mut v_idx = 0;

    let sides = [
        (
            [0.0, 0.0, 1.0],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ), // Front
        (
            [0.0, 0.0, -1.0],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ), // Back
        (
            [0.0, 1.0, 0.0],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ), // Top
        (
            [0.0, -1.0, 0.0],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
        ), // Bottom
        (
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ), // Right
        (
            [-1.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
        ), // Left
    ];

    for (normal, v0, v1, v2, v3) in sides {
        let n = [normal[0], normal[1], normal[2], 0.0];

        positions.push([v0[0], v0[1], v0[2], 1.0]);
        attributes.push(VertexAttributes {
            normal: n,
            uv: [0.0, 1.0, 0.0, 0.0],
        });

        positions.push([v1[0], v1[1], v1[2], 1.0]);
        attributes.push(VertexAttributes {
            normal: n,
            uv: [1.0, 1.0, 0.0, 0.0],
        });

        positions.push([v2[0], v2[1], v2[2], 1.0]);
        attributes.push(VertexAttributes {
            normal: n,
            uv: [1.0, 0.0, 0.0, 0.0],
        });

        positions.push([v3[0], v3[1], v3[2], 1.0]);
        attributes.push(VertexAttributes {
            normal: n,
            uv: [0.0, 0.0, 0.0, 0.0],
        });

        indices.push(v_idx);
        indices.push(v_idx + 1);
        indices.push(v_idx + 2);
        indices.push(v_idx);
        indices.push(v_idx + 2);
        indices.push(v_idx + 3);

        v_idx += 4;
    }

    build_blas(device, "Cube BLAS", positions, attributes, indices)
}

// --- ヘルパー関数: 球体(Sphere)のBLASを作成 (Icosphere) ---
pub fn create_sphere_blas(device: &wgpu::Device, subdivisions: u32) -> Geometry {
    let mut positions = Vec::new();
    let mut attributes = Vec::new();
    let mut indices = Vec::new();

    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    let mut add_vertex = |p: [f32; 3]| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let n = [p[0] / length, p[1] / length, p[2] / length];
        let pos = [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0];

        positions.push(pos);
        attributes.push(VertexAttributes {
            normal: [n[0], n[1], n[2], 0.0],
            uv: [0.0, 0.0, 0.0, 0.0], // TODO: Spherical mapping
        });
        positions.len() as u32 - 1
    };

    add_vertex([-1.0, t, 0.0]);
    add_vertex([1.0, t, 0.0]);
    add_vertex([-1.0, -t, 0.0]);
    add_vertex([1.0, -t, 0.0]);
    add_vertex([0.0, -1.0, t]);
    add_vertex([0.0, 1.0, t]);
    add_vertex([0.0, -1.0, -t]);
    add_vertex([0.0, 1.0, -t]);
    add_vertex([t, 0.0, -1.0]);
    add_vertex([t, 0.0, 1.0]);
    add_vertex([-t, 0.0, -1.0]);
    add_vertex([-t, 0.0, 1.0]);

    let mut faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut midpoint_cache = HashMap::new();
    for _ in 0..subdivisions {
        let mut new_faces = Vec::new();
        for tri in faces {
            let v1 = tri[0];
            let v2 = tri[1];
            let v3 = tri[2];

            let a = get_midpoint(v1, v2, &mut positions, &mut attributes, &mut midpoint_cache);
            let b = get_midpoint(v2, v3, &mut positions, &mut attributes, &mut midpoint_cache);
            let c = get_midpoint(v3, v1, &mut positions, &mut attributes, &mut midpoint_cache);

            new_faces.push([v1, a, c]);
            new_faces.push([v2, b, a]);
            new_faces.push([v3, c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }

    for tri in faces {
        indices.extend_from_slice(&tri);
    }

    build_blas(device, "Ico Sphere BLAS", positions, attributes, indices)
}

fn get_midpoint(
    p1: u32,
    p2: u32,
    positions: &mut Vec<[f32; 4]>,
    attributes: &mut Vec<VertexAttributes>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    if let Some(&index) = cache.get(&key) {
        return index;
    }

    let v1 = positions[p1 as usize];
    let v2 = positions[p2 as usize];

    let mid = [
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    ];

    let length = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    let n = [mid[0] / length, mid[1] / length, mid[2] / length];

    positions.push([n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0]);
    attributes.push(VertexAttributes {
        normal: [n[0], n[1], n[2], 0.0],
        uv: [0.0, 0.0, 0.0, 0.0],
    });

    let index = positions.len() as u32 - 1;
    cache.insert(key, index);
    index
}

use glam::Vec3;
use wgpu::util::DeviceExt;

pub fn create_crystal_blas(device: &wgpu::Device) -> Geometry {
    let mut positions = Vec::new();
    let mut attributes = Vec::new();
    let mut indices = Vec::new();

    // 頂点データ
    let top_tip = Vec3::new(0.0, 1.0, 0.0);
    let top_ring = [
        Vec3::new(0.3, 0.5, 0.3),
        Vec3::new(-0.3, 0.5, 0.3),
        Vec3::new(-0.3, 0.5, -0.3),
        Vec3::new(0.3, 0.5, -0.3),
    ];
    let bottom_ring = [
        Vec3::new(0.3, -0.5, 0.3),
        Vec3::new(-0.3, -0.5, 0.3),
        Vec3::new(-0.3, -0.5, -0.3),
        Vec3::new(0.3, -0.5, -0.3),
    ];
    let bottom_tip = Vec3::new(0.0, -1.0, 0.0);

    let mut add_face = |p0: Vec3, p1: Vec3, p2: Vec3| {
        // 法線を計算 (CCW winding points normal towards viewer)
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let n = edge1.cross(edge2).normalize();

        let base = positions.len() as u32;

        positions.push([p0.x, p0.y, p0.z, 1.0]);
        attributes.push(VertexAttributes {
            normal: [n.x, n.y, n.z, 0.0],
            uv: [0.0; 4],
        });

        positions.push([p1.x, p1.y, p1.z, 1.0]);
        attributes.push(VertexAttributes {
            normal: [n.x, n.y, n.z, 0.0],
            uv: [0.0; 4],
        });

        positions.push([p2.x, p2.y, p2.z, 1.0]);
        attributes.push(VertexAttributes {
            normal: [n.x, n.y, n.z, 0.0],
            uv: [0.0; 4],
        });

        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    };

    // 1. 上部四角錐 (4面) - CCW: tip -> next -> current
    for i in 0..4 {
        add_face(top_tip, top_ring[(i + 1) % 4], top_ring[i]);
    }

    // 2. 中間直方体 (4側面 * 2面)
    for i in 0..4 {
        let i_next = (i + 1) % 4;
        // Tri 1: top_current -> top_next -> bottom_next
        add_face(top_ring[i], top_ring[i_next], bottom_ring[i_next]);
        // Tri 2: top_current -> bottom_next -> bottom_current
        add_face(top_ring[i], bottom_ring[i_next], bottom_ring[i]);
    }

    // 3. 下部四角錐 (4面) - CCW: tip -> current -> next
    for i in 0..4 {
        add_face(bottom_tip, bottom_ring[i], bottom_ring[(i + 1) % 4]);
    }

    build_blas(
        device,
        "Refined Crystal BLAS",
        positions,
        attributes,
        indices,
    )
}
