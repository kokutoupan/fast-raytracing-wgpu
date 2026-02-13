pub mod builder;
pub mod light;
pub mod loader;
pub mod material;
pub mod resources;
pub mod scenes;

pub use material::Material;
pub use resources::SceneResources;

// src/scene/mod.rs に追加
pub const TEXTURE_WIDTH: u32 = 1024;
pub const TEXTURE_HEIGHT: u32 = 1024;
