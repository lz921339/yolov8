/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <opencv2/opencv.hpp>
#include <chrono>  // 添加时间测量库

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // 检查命令行参数数量，需要传入模型路径
    if (argc != 2)
    {
        printf("%s <model_path>\n", argv[0]);
        return -1;
    }

    // 获取模型路径参数
    const char *model_path = argv[1];

    int ret;
    // 初始化RKNN应用上下文结构体
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // 初始化后处理模块
    init_post_process();

    // 加载并初始化YOLOv8模型
    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    // 初始化源图像缓冲区和检测结果结构体
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    object_detect_result_list od_results;

    // 打开摄像头设备22号
    cv::VideoCapture cap(22);
    // 设置摄像头分辨率为640x480
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open the camera." << std::endl;
        return -1;
    }

    cv::Mat src_frame;
    // 创建名为"out"的OpenCV窗口
    cv::namedWindow("out", cv::WINDOW_NORMAL);
    // 设置窗口为全屏模式
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    
    // ===== FPS 统计变量 =====
    int frame_count = 0;
    double total_time = 0.0;
    double preprocess_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    auto fps_start_time = std::chrono::high_resolution_clock::now();
    
    // 主循环：持续读取摄像头帧并进行目标检测
    while(true){
        // 记录单帧开始时间
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // 从摄像头读取一帧图像
        cap >> src_frame;
        if (src_frame.empty()) {
            std::cerr << "Failed to grab frame from the camera." << std::endl;
            break;
        }

        // 将图像逆时针旋转90度
        cv::rotate(src_frame, src_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        
        // 填充源图像缓冲区信息
        src_image.height = src_frame.rows;
        src_image.width = src_frame.cols;
        src_image.width_stride = src_frame.step[0];  // 图像每行的字节数
        src_image.virt_addr = src_frame.data;        // 指向图像数据的指针
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        image_buffer_t dst_img;       // 目标图像缓冲区（用于模型输入）
        letterbox_t letter_box;       // letterbox变换参数
        rknn_input inputs[rknn_app_ctx.io_num.n_input];    // 模型输入
        rknn_output outputs[rknn_app_ctx.io_num.n_output]; // 模型输出
        const float nms_threshold = NMS_THRESH;             // 非极大值抑制阈值
        const float box_conf_threshold = BOX_THRESH;        // 检测框置信度阈值
        int bg_color = 0;  // 背景填充颜色

        // 初始化各结构体为0
        memset(&od_results, 0x00, sizeof(od_results));
        memset(&letter_box, 0, sizeof(letterbox_t));
        memset(&dst_img, 0, sizeof(image_buffer_t));
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));

        // ===== 预处理阶段 =====
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        
        // 设置目标图像尺寸为模型输入尺寸
        dst_img.width = rknn_app_ctx.model_width;
        dst_img.height = rknn_app_ctx.model_height;
        dst_img.format = IMAGE_FORMAT_RGB888;
        dst_img.size = get_image_size(&dst_img);
        
        // 为目标图像分配内存
        dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
        if (dst_img.virt_addr == NULL)
        {
            printf("malloc buffer size:%d fail!\n", dst_img.size);
            return -1;
        }

        // 使用letterbox方式进行图像缩放（保持宽高比，填充边缘）
        ret = convert_image_with_letterbox(&src_image, &dst_img, &letter_box, bg_color);
        if (ret < 0)
        {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
            return -1;
        }

        // ===== 设置模型输入 =====
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;  // 输入数据类型为uint8
        inputs[0].fmt = RKNN_TENSOR_NHWC;    // 数据格式：NHWC (batch, height, width, channel)
        inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
        inputs[0].buf = dst_img.virt_addr;   // 指向处理后的图像数据

        // 将输入数据传递给RKNN模型
        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        
        // 释放临时图像缓冲区
        free(dst_img.virt_addr);
        
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
        preprocess_time += preprocess_ms;
        
        // ===== 模型推理 =====
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // ===== 获取模型输出 =====
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++)
        {
            outputs[i].index = i;
            // 如果模型未量化，则获取浮点数输出；否则获取量化输出
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        inference_time += inference_ms;

        // ===== 后处理阶段 =====
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        
        // 对模型输出进行解析，应用NMS，生成最终检测结果
        post_process(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

        // 释放RKNN输出资源
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

        // ===== 绘制检测结果 =====
        char text[256];
        // 遍历所有检测到的目标
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // 获取边界框的四个坐标点
            int x1 = det_result->box.left;    // 左上角x坐标
            int y1 = det_result->box.top;     // 左上角y坐标
            int x2 = det_result->box.right;   // 右下角x坐标
            int y2 = det_result->box.bottom;  // 右下角y坐标

            // 在原图上绘制蓝色边界框，线宽为3像素
            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            // 格式化文本：类别名称 + 置信度百分比（保留1位小数）
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            
            // 在边界框上方绘制绿色文本标签，字体大小为10
            draw_text(&src_image, text, x1, y1 - 20, COLOR_GREEN, 10);
        }
        
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        postprocess_time += postprocess_ms;

        // 将处理后的图像数据封装为OpenCV Mat对象
        // CV_8UC3表示8位无符号3通道（BGR）图像
        cv::Mat result_mat = cv::Mat(src_image.height, src_image.width, CV_8UC3, 
                                     src_image.virt_addr, src_image.width_stride);

        // ===== 计算并显示 FPS =====
        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        total_time += frame_ms;
        frame_count++;
        
        // 每秒更新一次 FPS 统计
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - fps_start_time).count();
        
        if (elapsed >= 1.0) {
            double avg_fps = frame_count / elapsed;
            double avg_total = total_time / frame_count;
            double avg_preprocess = preprocess_time / frame_count;
            double avg_inference = inference_time / frame_count;
            double avg_postprocess = postprocess_time / frame_count;
            
            printf("\n========== Performance Statistics ==========\n");
            printf("FPS: %.2f\n", avg_fps);
            printf("Average Total Time: %.2f ms\n", avg_total);
            printf("  - Preprocess:  %.2f ms (%.1f%%)\n", avg_preprocess, avg_preprocess/avg_total*100);
            printf("  - Inference:   %.2f ms (%.1f%%)\n", avg_inference, avg_inference/avg_total*100);
            printf("  - Postprocess: %.2f ms (%.1f%%)\n", avg_postprocess, avg_postprocess/avg_total*100);
            printf("===========================================\n\n");
            
            // 重置统计
            frame_count = 0;
            total_time = 0.0;
            preprocess_time = 0.0;
            inference_time = 0.0;
            postprocess_time = 0.0;
            fps_start_time = current_time;
        }
        
        // 在图像上显示实时 FPS
        char fps_text[64];
        sprintf(fps_text, "FPS: %.1f", 1000.0 / frame_ms);
        draw_text(&src_image, fps_text, 10, 30, COLOR_RED, 15);

        // 显示检测结果图像
        cv::imshow("out", result_mat);

        // 等待1毫秒的键盘输入
        int key = cv::waitKey(1);
        // 如果按下'q'键或ESC键(ASCII 27)，则退出循环
        if (key == 'q' || key == 27) {
            break;
        }
    }
    
    // ===== 清理资源 =====
    // 释放后处理资源
    deinit_post_process();

    // 释放YOLOv8模型及相关资源
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    // 如果源图像缓冲区已分配内存，则释放（实际上这里src_image.virt_addr指向的是cv::Mat的数据，不需要手动释放）
    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
