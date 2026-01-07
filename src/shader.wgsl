enable wgpu_ray_query;

// --- 構造体定義 ---
struct Camera {
    view_inverse: array<vec4f, 4>,
    proj_inverse: array<vec4f, 4>,
    frame_count: u32,
}

struct Material {
    color: vec4f,
    emission: vec4f,
}

// --- バインドグループ ---
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read_write> accumulation: array<vec4f>;

// --- 乱数生成器 (PCG Hash) ---
// グローバル変数としてシードを持つ
var<private> rng_seed: u32;

fn init_rng(pos: vec2u, width: u32, frame: u32) {
    // 座標からシードを初期化
    rng_seed = pos.x + pos.y * width + frame * 927163u;
    // ちょっと回して散らす
    rng_seed = pcg_hash(rng_seed);
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 0.0 ～ 1.0 のランダムな float
fn rand() -> f32 {
    rng_seed = pcg_hash(rng_seed);
    return f32(rng_seed) / 4294967295.0;
}

// 単位球内のランダムなベクトル (拡散反射用)
fn random_in_unit_sphere() -> vec3f {
    for (var i = 0; i < 10; i++) {
        let p = vec3f(rand(), rand(), rand()) * 2.0 - 1.0;
        if length(p) < 1.0 {
            return p;
        }
    }
    return vec3f(0.0, 1.0, 0.0); // fallback
}

// --- レイ構造体 ---
struct Ray {
    origin: vec3f,
    dir: vec3f,
}

// --- メインの計算関数 ---
fn ray_color(r_in: Ray) -> vec3f {
    var r = r_in;
    var accumulated_color = vec3f(0.0); // 最終的な色
    var throughput = vec3f(1.0);        // 反射率の累積 (減衰)

    // バウンス回数 (Depth)
    let MAX_DEPTH = 5u;

    for (var i = 0u; i < MAX_DEPTH; i++) {
        var rq: ray_query;
        // 0.001 で自己交差を防ぐ (Shadow Acne対策)
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, r.origin, r.dir));
        rayQueryProceed(&rq);

        let hit = rayQueryGetCommittedIntersection(&rq);

        if hit.kind == 0u {
            // 何にも当たらなかったら背景色 (黒)
            // accumulated_color += vec3f(0.0) * throughput; 
            break;
        }

        // 1. マテリアル取得
        let mat_id = hit.instance_index;
        let mat = materials[mat_id];

        // 2. エミッションを加算
        // 光源に当たったら、その時点の throughput (反射率) を掛けて足す
        accumulated_color += mat.emission.rgb * throughput;

        // 3. 次のレイの準備 (法線計算)
        let obj_to_world = hit.object_to_world;
        let rotation_matrix = mat3x3f(obj_to_world[0], obj_to_world[1], obj_to_world[2]);
        let local_normal = vec3f(0.0, 1.0, 0.0); // Quadの法線
        let world_normal = normalize(rotation_matrix * local_normal);

        // 4. ランバート反射 (拡散反射)
        // 法線 + ランダムベクトル の方向に跳ね返る
        let target_dir = world_normal + random_in_unit_sphere();
        let new_dir = normalize(target_dir);

        // レイの更新
        // ヒット位置から少し浮かせた場所を新しい原点にする (t * dir)
        let hit_pos = r.origin + r.dir * hit.t;

        r.origin = hit_pos;
        r.dir = new_dir;

        // 5. 色の減衰
        // 壁の色を掛け合わせる (赤い壁なら、赤以外の光が吸収される)
        throughput *= mat.color.rgb;

        // ルーレット選択 (吸収されて暗くなったら打ち切る処理) を入れてもいいが、
        // 今回は単純な固定回数ループ
    }

    return accumulated_color;
}

// --- エントリーポイント ---
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_tex);
    if id.x >= size.x || id.y >= size.y { return; }

    // 乱数初期化
    init_rng(id.xy, size.x, camera.frame_count);

    let uv = vec2f(id.xy) / vec2f(size);
    // 元のNDC計算 (ジッターなしの参照用、ループ内で再計算する)
    // let ndc = vec2f(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);

    let view_inv = mat4x4f(camera.view_inverse[0], camera.view_inverse[1], camera.view_inverse[2], camera.view_inverse[3]);
    let proj_inv = mat4x4f(camera.proj_inverse[0], camera.proj_inverse[1], camera.proj_inverse[2], camera.proj_inverse[3]);
    let origin = view_inv[3].xyz;

    // 4x サンプリングループ
    var pixel_color_linear = vec3f(0.0);
    let samples_per_frame = 4u;

    for (var s = 0u; s < samples_per_frame; s++) {
        // サブピクセルジッター (0.0 ~ 1.0)
        let jitter = vec2f(rand(), rand());
        let uv_jittered = (vec2f(id.xy) + jitter) / vec2f(size);
        let ndc_jittered = vec2f(uv_jittered.x * 2.0 - 1.0, uv_jittered.y * 2.0 - 1.0);

        let target_ndc_jittered = vec4f(ndc_jittered, 1.0, 1.0);
        let target_world_jittered = view_inv * proj_inv * target_ndc_jittered;
        let direction_jittered = normalize(target_world_jittered.xyz / target_world_jittered.w - origin);

        let ray = Ray(origin, direction_jittered);
        pixel_color_linear += ray_color(ray);
    }

    // --- アキュムレーション処理 ---
    let idx = id.y * size.x + id.x;
    var current_acc = vec4f(0.0);
    if camera.frame_count > 0u {
        current_acc = accumulation[idx];
    }
    
    // 4サンプル分まとめて足す
    let new_acc = current_acc + vec4f(pixel_color_linear, f32(samples_per_frame));
    accumulation[idx] = new_acc;

    var final_color = new_acc.rgb / new_acc.w;

    // ガンマ補正 (線形空間 -> sRGB)
    // Reinhardトーンマップは一旦外して、単純なガンマ補正だけにする
    // (光が弱いと真っ黒に見えるのを防ぐため)
    final_color = pow(final_color, vec3f(1.0 / 2.2));

    textureStore(out_tex, id.xy, vec4f(final_color, 1.0));
}