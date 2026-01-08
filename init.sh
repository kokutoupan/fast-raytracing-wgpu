#!/bin/bash
set -e

# --- 設定 ---
OIDN_VER="2.3.3"
TAR_NAME="oidn-${OIDN_VER}.x86_64.linux.tar.gz"
DIR_NAME="oidn-${OIDN_VER}.x86_64.linux"
DOWNLOAD_URL="https://github.com/OpenImageDenoise/oidn/releases/download/v${OIDN_VER}/${TAR_NAME}"

# パス定義
PROJECT_ROOT="$(pwd)"
DEPS_DIR="${PROJECT_ROOT}/deps"
OIDN_ROOT="${DEPS_DIR}/${DIR_NAME}"
LIB_DIR="${OIDN_ROOT}/lib"
PKG_CONFIG_DIR="${LIB_DIR}/pkgconfig"
PC_FILE="${PKG_CONFIG_DIR}/OpenImageDenoise.pc"

# --- 1. OIDNのダウンロードと配置 ---
if [ ! -d "$OIDN_ROOT" ]; then
    echo "OIDN not found. Resetting deps..."
    rm -rf "$DEPS_DIR"
    mkdir -p "$DEPS_DIR"
    cd "$DEPS_DIR"

    echo "Downloading ${DOWNLOAD_URL}..."
    if command -v wget &> /dev/null; then
        wget -O "$TAR_NAME" "$DOWNLOAD_URL"
    else
        curl -L -o "$TAR_NAME" "$DOWNLOAD_URL"
    fi

    echo "Extracting..."
    tar -xf "$TAR_NAME"
    rm "$TAR_NAME"
    cd "$PROJECT_ROOT"
fi

# --- 2. .pc ファイルの強制生成 ---
echo "Generating .pc file..."
mkdir -p "$PKG_CONFIG_DIR"

cat <<EOF > "$PC_FILE"
prefix=${OIDN_ROOT}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: OpenImageDenoise
Description: Intel Open Image Denoise
Version: ${OIDN_VER}
Libs: -L\${libdir} -lOpenImageDenoise
Cflags: -I\${includedir}
EOF

# --- 3. ★追加: Cargo用設定ファイルの生成 ---
# これを作ると、VS Code(LSP)や通常のcargo runでもパスが通るようになります
CARGO_CONFIG_DIR="${PROJECT_ROOT}/.cargo"
CARGO_CONFIG_FILE="${CARGO_CONFIG_DIR}/config.toml"

echo "Generating Cargo config at: ${CARGO_CONFIG_FILE}"
mkdir -p "$CARGO_CONFIG_DIR"

cat <<EOF > "$CARGO_CONFIG_FILE"
# このファイルは run.sh によって自動生成されました。
# プロジェクト固有の環境変数を定義します。

[env]
# ビルド時に .pc ファイルを探すパス
PKG_CONFIG_PATH = { value = "${PKG_CONFIG_DIR}", force = true }

# 実行時に .so ライブラリを探すパス
LD_LIBRARY_PATH = { value = "${LIB_DIR}", force = true }
EOF

# --- 4. 実行 ---
echo "Setup complete. You can now use 'cargo run' directly."