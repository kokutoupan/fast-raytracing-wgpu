#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    pub type_: u32, // 0: Quad, 1: Sphere

    // Quadなら Uベクトル(横幅の半分), Sphereなら未定義
    pub u: [f32; 3],
    pub area: f32, // 確率計算(1/area)用

    // Quadなら Vベクトル(縦幅の半分), Sphereなら半径(radius)をxに入れる
    pub v: [f32; 3],
    pub pad: u32,

    pub emission: [f32; 4],
}
