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

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // ========== 1. 参数检查 ==========
    if (argc != 2)
    {
        printf("%s <model_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];  // 获取模型路径参数

    // ========== 2. 初始化 RKNN 上下文 ==========
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // 初始化后处理模块（加载标签等）
    init_post_process();

    // 加载并初始化 YOLOv8 模型
    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    // ========== 3. 准备图像缓冲区和结果容器 ==========
    image_buffer_t src_image;  // 源图像缓冲区（指向摄像头帧）
    memset(&src_image, 0, sizeof(image_buffer_t));
    object_detect_result_list od_results;  // 检测结果列表

    // ========== 4. 初始化摄像头 ==========
    cv::VideoCapture cap(22);  // 打开摄像头设备 /dev/video22
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);   // 设置宽度
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);  // 设置高度
    if (!cap.isOpened()) {
        std::cerr << "Failed to open the camera." << std::endl;
        return -1;
    }

    // ========== 5. 创建显示窗口 ==========
    cv::Mat original_highres, quick_frame;  // original_highres: 摄像头原始高分辨率图，quick_frame: 缩放到 640×480 的快速检测图
    cv::namedWindow("out", cv::WINDOW_NORMAL);
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // 尝试设置摄像头输出高分辨率（请求 3840×2160）
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 2160);
    int cam_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int cam_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("尝试设置摄像头分辨率 3840×2160，摄像头实际分辨率: %d × %d\n", cam_w, cam_h);

    // ========== 6. 主检测循环 ==========
    bool is_first_frame = true;  // 标记是否是第一帧
    
    while(true){
        // 6.1 从摄像头读取一帧（高分辨率原图）
        cap >> original_highres;
        if (original_highres.empty()) {
            std::cerr << "Failed to grab frame from the camera." << std::endl;
            break;
        }

        // 保存高分辨率的第一帧（旋转前）
        if (is_first_frame) {
            printf("========== 第一帧高分辨率信息 ==========%s\n", "");
            printf("高分辨率（旋转前）: %d × %d (宽×高)\n", original_highres.cols, original_highres.rows);
            std::string filename_high_before = "camera_first_frame_highres_before_rotate.jpg";
            if (cv::imwrite(filename_high_before, original_highres)) {
                printf("✓ 已保存高分辨率旋转前图像: %s\n", filename_high_before.c_str());
            } else {
                printf("✗ 保存高分辨率旋转前图像失败\n");
            }
        }

        // 6.2 图像预处理：旋转90度（适配摄像头安装方向）
        cv::rotate(original_highres, original_highres, cv::ROTATE_90_COUNTERCLOCKWISE);

        // 保存高分辨率的第一帧（旋转后）
        if (is_first_frame) {
            printf("高分辨率（旋转后）: %d × %d (宽×高)\n", original_highres.cols, original_highres.rows);
            std::string filename_high_after = "camera_first_frame_highres_after_rotate.jpg";
            if (cv::imwrite(filename_high_after, original_highres)) {
                printf("✓ 已保存高分辨率旋转后图像: %s\n", filename_high_after.c_str());
            } else {
                printf("✗ 保存高分辨率旋转后图像失败\n");
            }
        }

        // 6.3 生成用于快速检测的低分辨率图（先缩放到 640×480），随后会进行 letterbox 到 640×640
        cv::resize(original_highres, quick_frame, cv::Size(640, 480));

        // 保存 quick_frame 的第一帧用于调试
        if (is_first_frame) {
            std::string filename_quick = "camera_first_frame_quick_640x480.jpg";
            if (cv::imwrite(filename_quick, quick_frame)) {
                printf("✓ 已保存快速检测图: %s\n", filename_quick.c_str());
            } else {
                printf("✗ 保存快速检测图失败\n");
            }
            printf("===================================\n");
            is_first_frame = false;
        }

        // 下面使用 quick_frame 作为后续流程的 src_image 来源（指向 quick_frame 的数据）
        // 6.4 填充源图像结构体（指向 src_frame 的数据）
        src_image.height = quick_frame.rows;
        src_image.width = quick_frame.cols;
        src_image.width_stride = quick_frame.step[0];  // 每行字节数
        src_image.virt_addr = quick_frame.data;        // 图像数据指针
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = quick_frame.total() * quick_frame.elemSize();

        // 6.5 准备检测所需的变量
        int ret;
        image_buffer_t dst_img;        // 目标图像（640×640，用于模型输入）
        letterbox_t letter_box;        // Letterbox 变换参数（用于坐标映射）
        rknn_input inputs[rknn_app_ctx.io_num.n_input];    // RKNN 输入张量
        rknn_output outputs[rknn_app_ctx.io_num.n_output]; // RKNN 输出张量
        const float nms_threshold = NMS_THRESH;      // NMS 阈值 (0.45)
        const float box_conf_threshold = BOX_THRESH; // 置信度阈值 (0.25)
        int bg_color = 0;  // Letterbox 填充背景色（黑色）

        // 清零所有缓冲区
        memset(&od_results, 0x00, sizeof(od_results));
        memset(&letter_box, 0, sizeof(letterbox_t));
        memset(&dst_img, 0, sizeof(image_buffer_t));
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));

        // ========== 7. 图像预处理（Resize + Letterbox）==========
        // 7.1 分配目标图像缓冲区（640×640）
        dst_img.width = rknn_app_ctx.model_width;    // 模型输入宽度
        dst_img.height = rknn_app_ctx.model_height;  // 模型输入高度
        dst_img.format = IMAGE_FORMAT_RGB888;
        dst_img.size = get_image_size(&dst_img);
        dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
        if (dst_img.virt_addr == NULL)
        {
            printf("malloc buffer size:%d fail!\n", dst_img.size);
            return -1;
        }

        // 7.2 Letterbox 变换（保持宽高比，填充黑边）
        // 将 src_image 转换到 dst_img，记录变换参数到 letter_box
        ret = convert_image_with_letterbox(&src_image, &dst_img, &letter_box, bg_color);
        if (ret < 0)
        {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
            return -1;
        }

        // ========== 8. 设置模型输入 ==========
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;   // 输入类型：uint8
        inputs[0].fmt = RKNN_TENSOR_NHWC;     // 数据格式：NHWC
        inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
        inputs[0].buf = dst_img.virt_addr;    // 指向 letterbox 处理后的图像数据

        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        free(dst_img.virt_addr);  // 释放预处理图像缓冲区（数据已传入 RKNN）
        
        // ========== 9. 执行模型推理 ==========
        printf("rknn_run\n");
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // ========== 10. 获取模型输出 ==========
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++)
        {
            outputs[i].index = i;
            // 根据模型是否量化，决定输出格式（float 或 int8）
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);

        // ========== 11. 后处理（解析输出 + NMS）==========
        // 将模型输出解析为检测框，应用置信度过滤和 NMS
        // letter_box 用于将检测框坐标映射回原图
        post_process(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

        // 释放 RKNN 输出张量（必须释放，避免内存泄漏）
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

        // ========== 12. 绘制检测结果 ==========
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // 12.1 打印检测结果到控制台
            printf("%s @ (%d %d %d %d) %.3f\n",
                coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            
            // 12.2 获取边界框坐标
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            // 12.3 在原图上绘制蓝色边界框
            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            // 12.4 绘制类别名称和置信度文本（绿色）
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_GREEN, 10);
        }
        
        // ========== 13. 显示结果 ==========
        // 将 src_image 包装为 cv::Mat（无需拷贝数据）
        cv::Mat result_mat = cv::Mat(src_image.height, src_image.width, CV_8UC3, src_image.virt_addr, src_image.width_stride);
        cv::imshow("out", result_mat);
        
        // 检查按键输入（'q' 或 ESC 退出）
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    } // ========== 主循环结束 ==========
    
    // ========== 14. 清理资源 ==========
    // 14.1 反初始化后处理模块
    deinit_post_process();

    // 14.2 释放 YOLOv8 模型资源
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    // 14.3 释放图像缓冲区（如果有分配）
    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
// ========== 程序结束 ==========
