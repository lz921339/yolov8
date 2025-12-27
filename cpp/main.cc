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
    if (argc != 2)
    {
        printf("%s <model_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    object_detect_result_list od_results;

    cv::VideoCapture cap(22);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open the camera." << std::endl;
        return -1;
    }

    cv::Mat src_frame;
    cv::namedWindow("out", cv::WINDOW_NORMAL);
    cv::setWindowProperty("out", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    while(true){
        cap >> src_frame;
        if (src_frame.empty()) {
            std::cerr << "Failed to grab frame from the camera." << std::endl;
            break;
        }

        cv::rotate(src_frame, src_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        src_image.height = src_frame.rows;
        src_image.width = src_frame.cols;
        src_image.width_stride = src_frame.step[0];
        src_image.virt_addr = src_frame.data;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        int ret;
        image_buffer_t dst_img;
        letterbox_t letter_box;
        rknn_input inputs[rknn_app_ctx.io_num.n_input];
        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
        const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
        int bg_color = 0;

        memset(&od_results, 0x00, sizeof(od_results));
        memset(&letter_box, 0, sizeof(letterbox_t));
        memset(&dst_img, 0, sizeof(image_buffer_t));
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));

        // Pre Process
        dst_img.width = rknn_app_ctx.model_width;
        dst_img.height = rknn_app_ctx.model_height;
        dst_img.format = IMAGE_FORMAT_RGB888;
        dst_img.size = get_image_size(&dst_img);
        dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
        if (dst_img.virt_addr == NULL)
        {
            printf("malloc buffer size:%d fail!\n", dst_img.size);
            return -1;
        }

        // letterbox
        ret = convert_image_with_letterbox(&src_image, &dst_img, &letter_box, bg_color);
        if (ret < 0)
        {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
            return -1;
        }

        // Set Input Data
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
        inputs[0].buf = dst_img.virt_addr;

        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        free(dst_img.virt_addr);
        // Run
        printf("rknn_run\n");
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++)
        {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);


        // Post Process
        post_process(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

        // Remeber to release rknn output
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

        // 画框和概率
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n",
                coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_GREEN, 10);
        }
        cv::Mat result_mat = cv::Mat(src_image.height, src_image.width, CV_8UC3, src_image.virt_addr, src_image.width_stride);
        cv::imshow("out", result_mat);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }
    deinit_post_process();

    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
