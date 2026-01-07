enable wgpu_ray_query;

struct Camera {
    view_inverse: mat4x4f,
    proj_inverse: mat4x4f,
}
struct Material {
    color: vec4f,
    emission: vec4f,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage> materials: array<Material>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(out_tex);
    if id.x >= size.x || id.y >= size.y { return; }

    // 0.0 ~ 1.0 のUV
    let uv = vec2f(id.xy) / vec2f(size);
    // NDC座標 (-1.0 ~ 1.0) に変換 (Yは反転、プロジェクション行列次第)
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // --- レイの生成 (行列を使用) ---
    
    // レイの原点はカメラの位置（ビュー逆行列の平行移動成分）
    let origin: vec3f = camera.view_inverse[3].xyz;

    // レイのターゲット（スクリーン上の点）をワールド座標へ逆変換
    // Z=1.0 は遠方面、Z=0.0 は近方面 (wgpuのdepthは0-1)
    let target_ndc = vec4f(ndc.x, -ndc.y, 1.0, 1.0);
    let target_world = camera.view_inverse * camera.proj_inverse * target_ndc;
    
    // wで割って正規化 (Perspective除算の逆)
    let target_xyz = target_world.xyz / target_world.w;

    let direction = normalize(target_xyz - origin);

    // --- Ray Query 実行 ---
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, direction));
    rayQueryProceed(&rq);

    let hit = rayQueryGetCommittedIntersection(&rq);

    var color = vec3f(0.0);

    if hit.kind != 0u {
        let instance_id = hit.instance_index;

        let index = instance_id;
        let mat = materials[index];

        let world_pos = origin + direction * hit.t;
        // 3. ワールド法線の計算 (Normal)
        // object_to_world は mat4x3f (4列3行) です
        // 列0:X軸, 列1:Y軸, 列2:Z軸, 列3:平行移動
        let obj_to_world = hit.object_to_world;
        
        // 回転成分(3x3)だけ取り出す
        let rotation_matrix = mat3x3f(
            obj_to_world[0], // X軸ベクトル
            obj_to_world[1], // Y軸ベクトル
            obj_to_world[2]  // Z軸ベクトル
        );

        // Quadのローカル法線は (0, 1, 0)
        let local_normal = vec3f(0.0, 1.0, 0.0);
        
        // 回転行列を掛けてワールド法線にする
        let world_normal = normalize(rotation_matrix * local_normal);

        // 4. ランバート反射の計算 (Lambertian)
        // 天井のライト位置 (0.0, 1.9, 0.0) あたりに光源があると仮定
        let light_pos = vec3f(0.0, 1.9, 0.0);
        let light_vec = light_pos - world_pos;
        let light_dist = length(light_vec);
        let L = normalize(light_vec); // 光への方向ベクトル

        // 内積 (Cosine Law)
        // 面が光を向いているほど明るくなる (0.0未満はカット)
        let NdotL = max(abs(dot(world_normal, L)), 0.0);

        // 簡易的な距離減衰 (任意)
        let attenuation = 1.0 / (1.0 + light_dist * 0.5);

        // 5. 色の合成
        // 拡散反射光 (Diffuse) = マテリアル色 * 光の色(白) * 内積 * 減衰
        let diffuse = mat.color.rgb * vec3f(1.0) * NdotL * attenuation;
        
        // エミッション(発光)はそのまま加算
        color = diffuse + mat.emission.rgb;
    }
    // ガンマ補正 (sRGB出力用)
    color = color / (color + 1.0);
    color = pow(color / (color + 1.0), vec3f(1.0 / 2.2));

    textureStore(out_tex, vec2<i32>(id.xy), vec4f(color, 1.0));
}