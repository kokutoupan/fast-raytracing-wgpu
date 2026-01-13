use super::material::Material;
use super::resources::{MeshInfo, SceneResources};
use crate::geometry::{self, Vertex};
use glam::Mat4;
use wgpu::util::DeviceExt;

pub struct SceneBuilder {
    pub materials: Vec<Material>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub mesh_infos: Vec<MeshInfo>,
    pub instances: Vec<Option<wgpu::TlasInstance>>,
    pub blases: Vec<wgpu::Blas>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            mesh_infos: Vec::new(),
            instances: Vec::new(),
            blases: Vec::new(),
        }
    }

    pub fn add_material(&mut self, mat: Material) -> u32 {
        let id = self.materials.len() as u32;
        self.materials.push(mat);
        id
    }

    pub fn add_mesh(&mut self, geo: geometry::Geometry) -> u32 {
        let id = self.mesh_infos.len() as u32;

        let v_offset = self.vertices.len() as u32;
        let i_offset = self.indices.len() as u32;

        self.vertices.extend_from_slice(&geo.vertices);
        self.indices.extend_from_slice(&geo.indices);

        self.mesh_infos.push(MeshInfo {
            vertex_offset: v_offset,
            index_offset: i_offset,
            pad: [0; 2],
        });

        self.blases.push(geo.blas);

        id
    }

    pub fn add_instance(&mut self, mesh_id: u32, mat_id: u32, transform: Mat4) {
        let blas = &self.blases[mesh_id as usize];
        let affine = transform.transpose().to_cols_array();
        let instance_id = (mesh_id << 16) | mat_id;

        let instance =
            wgpu::TlasInstance::new(blas, affine[..12].try_into().unwrap(), instance_id, 0xff);
        self.instances.push(Some(instance));
    }

    pub fn build(mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
        let global_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let global_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Info Buffer"),
            contents: bytemuck::cast_slice(&self.mesh_infos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: bytemuck::cast_slice(&self.materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: self.instances.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        // Indexing into TLAS to set instances
        for (i, instance) in self.instances.drain(..).enumerate() {
            tlas[i] = instance;
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(None, Some(&tlas));
        queue.submit(std::iter::once(encoder.finish()));

        SceneResources {
            tlas,
            global_vertex_buffer,
            global_index_buffer,
            mesh_info_buffer,
            blases: self.blases,
            material_buffer,
        }
    }
}
