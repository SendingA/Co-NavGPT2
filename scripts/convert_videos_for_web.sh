#!/bin/bash
# =============================================================================
# 批量转换 MP4 视频为网页兼容格式
# 
# 问题: OpenCV 生成的 mp4v 编码视频在浏览器/GitHub Pages 无法播放
# 解决: 使用 ffmpeg 重新编码为 H.264 (libx264) + AAC
#
# 使用方法:
#     chmod +x scripts/convert_videos_for_web.sh
#     ./scripts/convert_videos_for_web.sh
#
#     # 指定输入目录
#     ./scripts/convert_videos_for_web.sh ./tmp/dump/gpt/videos
#
#     # 保留原文件
#     KEEP_ORIGINAL=1 ./scripts/convert_videos_for_web.sh
# =============================================================================

set -e

# 默认参数
INPUT_DIR="${1:-./tmp/dump/gpt/videos}"
KEEP_ORIGINAL="${KEEP_ORIGINAL:-0}"  # 0=删除原文件, 1=保留原文件
OUTPUT_SUFFIX="_web"  # 如果保留原文件，新文件的后缀

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   视频批量转换工具 (Web 兼容格式)${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查 ffmpeg 是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}错误: ffmpeg 未安装${NC}"
    echo "请先安装 ffmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  MacOS: brew install ffmpeg"
    echo "  Conda: conda install -c conda-forge ffmpeg"
    exit 1
fi

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 目录不存在: $INPUT_DIR${NC}"
    echo "用法: $0 [视频目录]"
    exit 1
fi

# 统计
TOTAL=0
SUCCESS=0
FAILED=0

echo -e "${YELLOW}输入目录: $INPUT_DIR${NC}"
echo -e "${YELLOW}保留原文件: $([ "$KEEP_ORIGINAL" = "1" ] && echo "是" || echo "否")${NC}"
echo ""

# 查找所有 mp4 文件
shopt -s nullglob
MP4_FILES=("$INPUT_DIR"/*.mp4)

if [ ${#MP4_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}未找到任何 .mp4 文件${NC}"
    exit 0
fi

echo -e "${GREEN}找到 ${#MP4_FILES[@]} 个视频文件${NC}"
echo ""

# 处理每个视频
for input_file in "${MP4_FILES[@]}"; do
    TOTAL=$((TOTAL + 1))
    filename=$(basename "$input_file")
    
    # 跳过已经转换过的文件
    if [[ "$filename" == *"${OUTPUT_SUFFIX}.mp4" ]]; then
        echo -e "${YELLOW}跳过已转换: $filename${NC}"
        continue
    fi
    
    echo -e "${BLUE}[$TOTAL/${#MP4_FILES[@]}] 处理: $filename${NC}"
    
    if [ "$KEEP_ORIGINAL" = "1" ]; then
        # 保留原文件，生成新文件
        output_file="${input_file%.mp4}${OUTPUT_SUFFIX}.mp4"
    else
        # 替换原文件
        output_file="${input_file%.mp4}_temp.mp4"
    fi
    
    # ffmpeg 转换命令
    # -y: 覆盖输出文件
    # -i: 输入文件
    # -c:v libopenh264: 使用 OpenH264 编码 (libx264 的替代)
    # -pix_fmt yuv420p: 像素格式，确保浏览器兼容
    # -movflags +faststart: 将 moov atom 放在文件开头，支持流式播放
    # -an: 无音频（我们的视频本来就没音频）
    
    if ffmpeg -y -i "$input_file" \
        -c:v libopenh264 \
        -pix_fmt yuv420p \
        -movflags +faststart \
        -an \
        "$output_file" \
        -loglevel warning 2>&1; then
        
        if [ "$KEEP_ORIGINAL" = "0" ]; then
            # 替换原文件
            mv "$output_file" "$input_file"
        fi
        
        echo -e "${GREEN}  ✓ 转换成功${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}  ✗ 转换失败${NC}"
        FAILED=$((FAILED + 1))
        # 删除失败的临时文件
        [ -f "$output_file" ] && rm "$output_file"
    fi
done

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}转换完成!${NC}"
echo -e "  总计: $TOTAL"
echo -e "  成功: ${GREEN}$SUCCESS${NC}"
echo -e "  失败: ${RED}$FAILED${NC}"
echo -e "${BLUE}============================================${NC}"

if [ "$KEEP_ORIGINAL" = "1" ]; then
    echo -e "${YELLOW}提示: 新文件以 '${OUTPUT_SUFFIX}.mp4' 结尾${NC}"
fi
