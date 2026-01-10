use std::collections::HashMap;

// 頂点データ (32バイト)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 4],    // vec4f alignment
    pub normal: [f32; 4], // vec4f alignment
}

pub struct Geometry {
    pub blas: wgpu::Blas,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub desc: wgpu::BlasTriangleGeometrySizeDescriptor,
}

fn build_blas(
    device: &wgpu::Device,
    label: &str,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
) -> Geometry {
    let desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertices.len() as u32,
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
        vertices,
        indices,
        desc,
    }
}

// --- ヘルパー関数: 平面(Quad)のBLASを作成 ---
pub fn create_plane_blas(device: &wgpu::Device) -> Geometry {
    // 1x1 の平面 (XZ平面, 中心0,0)
    let vertices = vec![
        Vertex {
            pos: [-0.5, 0.0, 0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 左手前
        Vertex {
            pos: [0.5, 0.0, 0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 右手前
        Vertex {
            pos: [-0.5, 0.0, -0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 左奥
        Vertex {
            pos: [0.5, 0.0, -0.5, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
        }, // 右奥
    ];
    let indices: Vec<u32> = vec![0, 1, 2, 2, 1, 3]; // Triangle List

    build_blas(device, "Quad BLAS", vertices, indices)
}

// --- ヘルパー関数: 立方体(Cube)のBLASを作成 ---
pub fn create_cube_blas(device: &wgpu::Device) -> Geometry {
    let mut vertices = Vec::new();
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
        vertices.push(Vertex {
            pos: [v0[0], v0[1], v0[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v1[0], v1[1], v1[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v2[0], v2[1], v2[2], 1.0],
            normal: n,
        });
        vertices.push(Vertex {
            pos: [v3[0], v3[1], v3[2], 1.0],
            normal: n,
        });

        indices.push(v_idx);
        indices.push(v_idx + 1);
        indices.push(v_idx + 2);
        indices.push(v_idx);
        indices.push(v_idx + 2);
        indices.push(v_idx + 3);

        v_idx += 4;
    }

    build_blas(device, "Cube BLAS", vertices, indices)
}

// --- ヘルパー関数: 球体(Sphere)のBLASを作成 (Icosphere) ---
pub fn create_sphere_blas(device: &wgpu::Device, subdivisions: u32) -> Geometry {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    let mut add_vertex = |p: [f32; 3]| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let n = [p[0] / length, p[1] / length, p[2] / length];
        let pos = [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0];
        vertices.push(Vertex {
            pos,
            normal: [n[0], n[1], n[2], 0.0],
        });
        vertices.len() as u32 - 1
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

            let a = get_midpoint(v1, v2, &mut vertices, &mut midpoint_cache);
            let b = get_midpoint(v2, v3, &mut vertices, &mut midpoint_cache);
            let c = get_midpoint(v3, v1, &mut vertices, &mut midpoint_cache);

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

    build_blas(device, "Ico Sphere BLAS", vertices, indices)
}

fn get_midpoint(
    p1: u32,
    p2: u32,
    vertices: &mut Vec<Vertex>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    if let Some(&index) = cache.get(&key) {
        return index;
    }

    let v1 = vertices[p1 as usize].pos;
    let v2 = vertices[p2 as usize].pos;

    let mid = [
        (v1[0] + v2[0]) * 0.5,
        (v1[1] + v2[1]) * 0.5,
        (v1[2] + v2[2]) * 0.5,
    ];

    let length = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    let n = [mid[0] / length, mid[1] / length, mid[2] / length];

    vertices.push(Vertex {
        pos: [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5, 1.0],
        normal: [n[0], n[1], n[2], 0.0],
    });

    let index = vertices.len() as u32 - 1;
    cache.insert(key, index);
    index
}
