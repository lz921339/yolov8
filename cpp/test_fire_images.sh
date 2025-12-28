#!/bin/bash

# 火焰检测测试脚本
# 使用方法: ./test_fire_images.sh

echo "🔥 火焰检测测试开始..."
echo "================================"

# 检查可执行文件是否存在
if [ ! -f "./rknn_yolov8_cam" ]; then
    echo "❌ 错误: 找不到 rknn_yolov8_cam 可执行文件"
    echo "请先编译: ./build-linux.sh"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "./model/yolov8.rknn" ]; then
    echo "❌ 错误: 找不到 yolov8.rknn 模型文件"
    echo "请将火焰检测模型重命名为 yolov8.rknn 并放在 model/ 目录下"
    exit 1
fi

# 测试图片列表
images=("1.jpg" "2.jpg" "3.jpg")

echo "📸 开始测试火焰检测图片..."
echo ""

# 测试每张图片
for img in "${images[@]}"; do
    if [ -f "./model/$img" ]; then
        echo "🔍 测试图片: $img"
        echo "--------------------------------"
        
        # 使用摄像头程序测试单张图片 (需要修改程序支持图片输入)
        echo "⚠️  注意: 当前程序是摄像头版本，需要修改支持图片输入"
        echo "建议使用静态图片检测版本进行测试"
        echo ""
    else
        echo "⚠️  警告: 找不到测试图片 ./model/$img"
    fi
done

echo "📋 测试图片列表:"
for img in "${images[@]}"; do
    if [ -f "./model/$img" ]; then
        echo "✅ $img - 存在"
    else
        echo "❌ $img - 不存在"
    fi
done

echo ""
echo "🎯 使用说明:"
echo "1. 当前程序是摄像头实时检测版本"
echo "2. 如需测试静态图片，建议使用图片检测版本"
echo "3. 摄像头版本使用方法: ./rknn_yolov8_cam ./model/yolov8.rknn"
echo ""
echo "🎉 火焰检测测试脚本完成！"
