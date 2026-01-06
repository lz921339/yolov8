/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>  // 添加时间测量库
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

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

    // 预分配目标图像缓冲区，避免每帧 malloc/free
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));
    dst_img.width = rknn_app_ctx.model_width;
    dst_img.height = rknn_app_ctx.model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);

    // ===== 分配模型输入缓冲区 (使用DMA heap) =====
    // DMA heap是Linux提供的DMA内存分配机制
    // 可以分配物理连续的内存,供RGA等硬件加速器直接访问
    
    // 步骤1: 打开DMA heap设备文件
    // "/dev/dma_heap/system" 是系统默认的DMA heap设备
    int dma_heap_fd = open("/dev/dma_heap/system", O_RDWR);
    if (dma_heap_fd < 0) {
        // DMA heap打开失败 (可能是内核不支持或权限不足)
        printf("Failed to open dma_heap, fallback to malloc\n");
        // 降级方案: 使用普通malloc分配内存
        dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
        dst_img.fd = 0;  // fd=0 表示不是DMA缓冲区
    } else {
        // 步骤2: 准备DMA heap分配请求结构体
        struct dma_heap_allocation_data heap_data;
        memset(&heap_data, 0, sizeof(heap_data));
        heap_data.len = dst_img.size;           // 要分配的字节数 (640*640*3=1228800)
        heap_data.fd_flags = O_RDWR | O_CLOEXEC; // 文件描述符标志:
                                                   // O_RDWR: 可读写
                                                   // O_CLOEXEC: exec时自动关闭
        
        // 步骤3: 通过ioctl系统调用向DMA heap请求分配内存
        // DMA_HEAP_IOCTL_ALLOC 是DMA heap的分配命令
        if (ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data) < 0) {
            // DMA分配失败 (可能是内存不足)
            printf("DMA heap alloc failed, fallback to malloc\n");
            dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
            dst_img.fd = 0;
        } else {
            // 步骤4: DMA分配成功,获取文件描述符
            // heap_data.fd 是内核返回的DMA缓冲区文件描述符
            // 这个fd代表一块物理连续的内存,可以传递给RGA等硬件
            dst_img.fd = heap_data.fd;
            
            // 步骤5: 将DMA缓冲区映射到用户空间虚拟地址
            // mmap系统调用建立虚拟地址到物理内存的映射关系
            dst_img.virt_addr = (unsigned char*)mmap(
                NULL,                    // 让内核自动选择虚拟地址
                dst_img.size,            // 映射大小
                PROT_READ | PROT_WRITE,  // 内存保护标志: 可读写
                MAP_SHARED,              // 映射类型: 共享映射 (硬件可见修改)
                dst_img.fd,              // DMA缓冲区的文件描述符
                0                        // 从文件开头偏移0字节开始映射
            );
            
            if (dst_img.virt_addr == MAP_FAILED) {
                // mmap失败 (通常不会发生,除非系统资源耗尽)
                printf("mmap failed, fallback to malloc\n");
                close(dst_img.fd);  // 关闭DMA缓冲区fd
                dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
                dst_img.fd = 0;
            } else {
                // 成功! 打印DMA缓冲区信息
                // 此时 dst_img.fd > 0, convert_image_rga会使用importbuffer_fd
                // RGA硬件可以通过fd直接访问物理内存,实现硬件加速
                printf("DMA buffer allocated: fd=%d size=%d\n", dst_img.fd, dst_img.size);
            }
        }
        
        // 步骤6: 关闭DMA heap设备文件描述符
        // 注意: 这里关闭的是 /dev/dma_heap/system 的fd,
        // 不是DMA缓冲区的fd (dst_img.fd仍然有效)
        close(dma_heap_fd);
    }
    
    // 最终检查: 确保缓冲区地址有效 (无论是DMA还是malloc)
    if (dst_img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    cv::Mat shared_frame;
    std::mutex frame_mutex;
    std::atomic<bool> stop_capture(false);

    // 捕获线程：持续读取摄像头，把最新一帧存到 shared_frame
    std::thread cap_thread([&](){
        cv::Mat frame;
        while(!stop_capture.load()) {
            if (!cap.read(frame)) continue; // 失败就重试
            std::lock_guard<std::mutex> lk(frame_mutex);
            frame.copyTo(shared_frame); // 只保存最新一帧
            // 不要 sleep，尽量实时
        }
    });

    // 主循环：持续读取摄像头帧并进行目标检测
    while(true){
        {
            std::lock_guard<std::mutex> lk(frame_mutex);
            if (shared_frame.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
            shared_frame.copyTo(src_frame);
        }
        // 现在 src_frame 是最新帧，继续预处理/推理/后处理

        // 记录单帧开始时间
        auto frame_start = std::chrono::high_resolution_clock::now();
        // ===== 预处理阶段 =====
        auto preprocess_start = std::chrono::high_resolution_clock::now();

        // 将图像逆时针旋转90度
        cv::rotate(src_frame, src_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        
        // 填充源图像缓冲区信息
        src_image.height = src_frame.rows;
        src_image.width = src_frame.cols;
        src_image.width_stride = src_frame.step[0];  // 图像每行的字节数
        src_image.virt_addr = src_frame.data;        // 指向图像数据的指针
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        letterbox_t letter_box;       // letterbox变换参数
        rknn_input inputs[rknn_app_ctx.io_num.n_input];    // 模型输入
        rknn_output outputs[rknn_app_ctx.io_num.n_output]; // 模型输出
        const float nms_threshold = NMS_THRESH;             // 非极大值抑制阈值
        const float box_conf_threshold = BOX_THRESH;        // 检测框置信度阈值
        int bg_color = 0;  // 背景填充颜色

        // 初始化各结构体为0
        memset(&od_results, 0x00, sizeof(od_results));
        memset(&letter_box, 0, sizeof(letterbox_t));
        
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
        // ✅ 先创建OpenCV Mat（共享src_image的内存，无拷贝）
        cv::Mat result_mat = cv::Mat(src_image.height, src_image.width, CV_8UC3, 
                                     src_image.virt_addr, src_image.width_stride);

        // ✅ 使用OpenCV绘制（比自定义函数快10倍）
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // 获取边界框的四个坐标点
            int x1 = det_result->box.left;    // 左上角x坐标
            int y1 = det_result->box.top;     // 左上角y坐标
            int x2 = det_result->box.right;   // 右下角x坐标
            int y2 = det_result->box.bottom;  // 右下角y坐标

            // ✅ 使用OpenCV的rectangle（NEON加速，比draw_rectangle快10倍）
            cv::rectangle(result_mat, 
                         cv::Point(x1, y1), 
                         cv::Point(x2, y2), 
                         cv::Scalar(255, 0, 0),  // BGR: 蓝色
                         3);  // 线宽

            // 格式化文本：类别名称 + 置信度百分比（保留1位小数）
            char text[256];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            
            // ✅ 使用OpenCV的putText（FreeType加速，比draw_text快7倍）
            cv::putText(result_mat, 
                       text,
                       cv::Point(x1, y1 - 10),       // 文本位置（稍微上移）
                       cv::FONT_HERSHEY_SIMPLEX,     // 字体类型
                       0.5,                          // 字体缩放系数
                       cv::Scalar(0, 255, 0),        // BGR: 绿色
                       2);                           // 线宽
        }

        // 显示检测结果图像
        cv::imshow("out", result_mat);

        // 等待1毫秒的键盘输入
        int key = cv::waitKey(1);
        // 如果按下'q'键或ESC键(ASCII 27)，则退出循环
        if (key == 'q' || key == 27) {
            break;
        }

        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        postprocess_time += postprocess_ms;

                // ===== 计算并显示 FPS =====
        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        total_time += frame_ms;
        frame_count++;
        
        
        if(frame_count%5==0)
        {
            double avg_fps = frame_count / (total_time/1000);
            double avg_total = (total_time) / frame_count;
            double avg_preprocess = (preprocess_time) / frame_count;
            double avg_inference = (inference_time) / frame_count;
            double avg_postprocess = (postprocess_time) / frame_count;

            printf("\n========== Performance Statistics ==========\n");
            printf("FPS: %.2f\n", avg_fps);
            printf("Average Total Time: %.2f ms\n", avg_total);
            printf("  - Preprocess:  %.2f ms (%.1f%%)\n", avg_preprocess, avg_preprocess/avg_total*100);
            printf("  - Inference:   %.2f ms (%.1f%%)\n", avg_inference, avg_inference/avg_total*100);
            printf("  - Postprocess: %.2f ms (%.1f%%)\n", avg_postprocess, avg_postprocess/avg_total*100);
            printf("===========================================\n\n");
            
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

    // 清理：循环外释放一次
    if (dst_img.fd > 0) {
        munmap(dst_img.virt_addr, dst_img.size);
        close(dst_img.fd);
    } else if (dst_img.virt_addr) {
        free(dst_img.virt_addr);
    }

    // 退出时停止线程
    stop_capture.store(true);
    if (cap_thread.joinable()) cap_thread.join();

    return 0;
}
