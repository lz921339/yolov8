#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <sys/time.h>

#include "im2d.h"
#include "drmrga.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "turbojpeg.h"

#include "image_utils.h"
#include "file_utils.h"

static const char* filter_image_names[] = {
    "jpg",
    "jpeg",
    "JPG",
    "JPEG",
    "png",
    "PNG",
    "data",
    NULL
};

static const char* subsampName[TJ_NUMSAMP] = {"4:4:4", "4:2:2", "4:2:0", "Grayscale", "4:4:0", "4:1:1"};

static const char* colorspaceName[TJ_NUMCS] = {"RGB", "YCbCr", "GRAY", "CMYK", "YCCK"};

static int image_file_filter(const struct dirent *entry)
{
    const char ** filter;

    for (filter = filter_image_names; *filter; ++filter) {
        if(strstr(entry->d_name, *filter) != NULL) {
            return 1;
        }
    }
    return 0;
}

static int read_image_jpeg(const char* path, image_buffer_t* image)
{
    FILE* jpegFile = NULL;
    unsigned long jpegSize;
    int flags = 0;
    int width, height;
    int origin_width, origin_height;
    unsigned char* imgBuf = NULL;
    unsigned char* jpegBuf = NULL;
    unsigned long size;
    unsigned short orientation = 1;
    struct timeval tv1, tv2;

    if ((jpegFile = fopen(path, "rb")) == NULL) {
        printf("open input file failure\n");
    }
    if (fseek(jpegFile, 0, SEEK_END) < 0 || (size = ftell(jpegFile)) < 0 || fseek(jpegFile, 0, SEEK_SET) < 0) {
        printf("determining input file size failure\n");
    }
    if (size == 0) {
        printf("determining input file size, Input file contains no data\n");
    }
    jpegSize = (unsigned long)size;
    if ((jpegBuf = (unsigned char*)malloc(jpegSize * sizeof(unsigned char))) == NULL) {
        printf("allocating JPEG buffer\n");
    }
    if (fread(jpegBuf, jpegSize, 1, jpegFile) < 1) {
        printf("reading input file");
    }
    fclose(jpegFile);
    jpegFile = NULL;

    tjhandle handle = NULL;
    int subsample, colorspace;
    int padding = 1;
    int ret = 0;

    handle = tjInitDecompress();
    ret = tjDecompressHeader3(handle, jpegBuf, size, &origin_width, &origin_height, &subsample, &colorspace);
    if (ret < 0) {
        printf("header file error, errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
        return -1;
    }

    // 对图像做裁剪16对齐，利于后续rga操作
    int crop_width = origin_width / 16 * 16;
    int crop_height = origin_height / 16 * 16;

    printf("origin size=%dx%d crop size=%dx%d\n", origin_width, origin_height, crop_width, crop_height);

    // gettimeofday(&tv1, NULL);
    ret = tjDecompressHeader3(handle, jpegBuf, size, &width, &height, &subsample, &colorspace);
    if (ret < 0) {
        printf("header file error, errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
        return -1;
    }
    printf("input image: %d x %d, subsampling: %s, colorspace: %s, orientation: %d\n", 
            width, height, subsampName[subsample], colorspaceName[colorspace], orientation);
    int sw_out_size = width * height * 3;
    unsigned char* sw_out_buf = image->virt_addr;
    if (sw_out_buf == NULL) {
        sw_out_buf = (unsigned char*)malloc(sw_out_size * sizeof(unsigned char));
    }
    if (sw_out_buf == NULL) {
        printf("sw_out_buf is NULL\n");
        goto out;
    }

    flags |= 0;

    // 错误码为0时，表示警告，错误码为-1时表示错误
    int pixelFormat = TJPF_RGB;
    ret = tjDecompress2(handle, jpegBuf, size, sw_out_buf, width, 0, height, pixelFormat, flags);
    // ret = tjDecompressToYUV2(handle, jpeg_buf, size, dst_buf, *width, padding, *height, flags);
    if ((0 != tjGetErrorCode(handle)) && (ret < 0)) {
        printf("error : decompress to yuv failed, errorStr:%s, errorCode:%d\n", tjGetErrorStr(),
               tjGetErrorCode(handle));
        goto out;
    }
    if ((0 == tjGetErrorCode(handle)) && (ret < 0)) {
        printf("warning : errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
    }
    tjDestroy(handle);
    // gettimeofday(&tv2, NULL);
    // printf("decode time %ld ms\n", (tv2.tv_sec-tv1.tv_sec)*1000 + (tv2.tv_usec-tv1.tv_usec)/1000);

    image->width = width;
    image->height = height;
    image->format = IMAGE_FORMAT_RGB888;
    image->virt_addr = sw_out_buf;
    image->size = sw_out_size;
out:
    if (jpegBuf) {
        free(jpegBuf);
    }
    return 0;
}

static int read_image_raw(const char* path, image_buffer_t* image)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    unsigned char *data = image->virt_addr;
    if (image->virt_addr == NULL) {
        data = (unsigned char *)malloc(file_size+1);
    }
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    if (image->virt_addr == NULL) {
        image->virt_addr = data;
        image->size = file_size;
    }

    return 0;
}

static int write_image_jpeg(const char* path, int quality, const image_buffer_t* image)
{
    int ret;
    int jpegSubsamp = TJSAMP_422;
    unsigned char* jpegBuf = NULL;
    unsigned long jpegSize = 0;
    int flags = 0;

    const unsigned char* data = image->virt_addr;
    int width = image->width;
    int height = image->height;
    int pixelFormat = TJPF_RGB;

	tjhandle handle = tjInitCompress();

    if (image->format == IMAGE_FORMAT_RGB888) {
        ret = tjCompress2(handle, data, width, 0, height, pixelFormat, &jpegBuf, &jpegSize, jpegSubsamp, quality, flags);
    } else {
        printf("write_image_jpeg: pixel format %d not support\n", image->format);
        return -1;
    }

	// printf("ret=%d jpegBuf=%p jpegSize=%d\n", ret, jpegBuf, jpegSize);
    if (jpegBuf != NULL && jpegSize > 0) {
        write_data_to_file(path, (const char*)jpegBuf, jpegSize);
        tjFree(jpegBuf);
    }
    tjDestroy(handle);

	return 0;
}

static int read_image_stb(const char* path, image_buffer_t* image)
{
    // 默认图像为3通道
    int w, h, c;
    unsigned char* pixeldata = stbi_load(path, &w, &h, &c, 0);
    if (!pixeldata) {
        printf("error: read image %s fail\n", path);
        return -1;
    }
    // printf("load image wxhxc=%dx%dx%d path=%s\n", w, h, c, path);
    int size = w * h * c;

    // 设置图像数据
    if (image->virt_addr != NULL) {
        memcpy(image->virt_addr, pixeldata, size);
        stbi_image_free(pixeldata);
    } else {
        image->virt_addr = pixeldata;
    }
    image->width = w;
    image->height = h;
    if (c == 4) {
        image->format = IMAGE_FORMAT_RGBA8888;
    } else if (c == 1) {
        image->format = IMAGE_FORMAT_GRAY8;
    } else {
        image->format = IMAGE_FORMAT_RGB888;
    }
    return 0;
}

int read_image(const char* path, image_buffer_t* image)
{
    const char* _ext = strrchr(path, '.');
    if (!_ext) {
        // missing extension
        return -1;
    }
    if (strcmp(_ext, ".data") == 0) {
        return read_image_raw(path, image);
    } else if (strcmp(_ext, ".jpg") == 0 || strcmp(_ext, ".jpeg") == 0 || strcmp(_ext, ".JPG") == 0 ||
        strcmp(_ext, ".JPEG") == 0) {
        return read_image_jpeg(path, image);
    } else {
        return read_image_stb(path, image);
    }
}

int write_image(const char* path, const image_buffer_t* img)
{
    int ret;
    int width = img->width;
    int height = img->height;
    int channel = 3;
    void* data = img->virt_addr;
    printf("write_image path: %s width=%d height=%d channel=%d data=%p\n",
        path, width, height, channel, data);

    const char* _ext = strrchr(path, '.');
    if (!_ext) {
        // missing extension
        return -1;
    }
    if (strcmp(_ext, ".jpg") == 0 || strcmp(_ext, ".jpeg") == 0 || strcmp(_ext, ".JPG") == 0 ||
        strcmp(_ext, ".JPEG") == 0) {
        int quality = 95;
        ret = write_image_jpeg(path, quality, img);
    } else if (strcmp(_ext, ".png") == 0 | strcmp(_ext, ".PNG") == 0) {
        ret = stbi_write_png(path, width, height, channel, data, 0);
    } else if (strcmp(_ext, ".data") == 0 | strcmp(_ext, ".DATA") == 0) {
        int size = get_image_size(img);
        ret = write_data_to_file(path, data, size);
    } else {
        // unknown extension type
        return -1;
    }
    return ret;
}

static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
    if (dst == NULL) {
        printf("dst buffer is null\n");
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
    //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
    // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
    //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

    // 从原图指定区域取数据，双线性缩放到目标指定区域
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;    // down
            if (src_y == src_height - 1) {
                // 如果到图像最下边缘，变成选择上面的像素
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;            // right
            int index4 = index2 + 1 * channel;            // down right
            if (src_x == src_width - 1) {
                // 如果到图像最右边缘，变成选择左边的像素
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
            //     dst_x, dst_y, dst_x_offset, dst_y_offset,
            //     src_x, src_y, x_diff, y_diff,
            //     index1, index2, index3, index4);

            for (int c = 0; c < channel; c++) {
                unsigned char A = src[index1+c];
                unsigned char B = src[index3+c];
                unsigned char C = src[index2+c];
                unsigned char D = src[index4+c];

                unsigned char pixel = (unsigned char)(
                    A * (1 - x_diff) * (1 - y_diff) +
                    B * x_diff * (1 - y_diff) +
                    C * y_diff * (1 - x_diff) +
                    D * x_diff * y_diff
                );

                dst[(dst_y * dst_width  + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}

static int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {

    unsigned char* src_y = src;
    unsigned char* src_uv = src + src_width * src_height;

    unsigned char* dst_y = dst;
    unsigned char* dst_uv = dst + dst_width * dst_height;

    crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height,
        dst_y, dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    
    crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2, crop_height / 2,
        dst_uv, dst_width / 2, dst_height / 2, dst_box_x, dst_box_y, dst_box_width, dst_box_height);

    return 0;
}

static int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst, image_rect_t *src_box, image_rect_t *dst_box, char color) {
    int ret;
    if (dst->virt_addr == NULL) {
        return -1;
    }
    if (src->virt_addr == NULL) {
        return -1;
    }
    if (src->format != dst->format) {
        return -1;
    }

    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = src->width;
    int src_box_h = src->height;
    if (src_box != NULL) {
        src_box_x = src_box->left;
        src_box_y = src_box->top;
        src_box_w = src_box->right - src_box->left + 1;
        src_box_h = src_box->bottom - src_box->top + 1;
    }
    int dst_box_x = 0;
    int dst_box_y = 0;
    int dst_box_w = dst->width;
    int dst_box_h = dst->height;
    if (dst_box != NULL) {
        dst_box_x = dst_box->left;
        dst_box_y = dst_box->top;
        dst_box_w = dst_box->right - dst_box->left + 1;
        dst_box_h = dst_box->bottom - dst_box->top + 1;
    }

    // fill pad color
    if (dst_box_w != dst->width || dst_box_h != dst->height) {
        int dst_size = get_image_size(dst);
        memset(dst->virt_addr, color, dst_size);
    }

    int need_release_dst_buffer = 0;
    int reti = 0;
    if (src->format == IMAGE_FORMAT_RGB888) {
        reti = crop_and_scale_image_c(3, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_RGBA8888) {
        reti = crop_and_scale_image_c(4, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_GRAY8) {
        reti = crop_and_scale_image_c(1, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_YUV420SP_NV12 || src->format == IMAGE_FORMAT_YUV420SP_NV21) {
        reti = crop_and_scale_image_yuv420sp(src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else {
        printf("no support format %d\n", src->format);
    }
    if (reti != 0) {
        printf("convert_image_cpu fail %d\n", reti);
        return -1;
    }
    printf("finish\n");
    return 0;
}

static int get_rga_fmt(image_format_t fmt) {
    switch (fmt)
    {
    case IMAGE_FORMAT_RGB888:
        return RK_FORMAT_RGB_888;
    case IMAGE_FORMAT_RGBA8888:
        return RK_FORMAT_RGBA_8888;
    case IMAGE_FORMAT_YUV420SP_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case IMAGE_FORMAT_YUV420SP_NV21:
        return RK_FORMAT_YCrCb_420_SP;
    default:
        return -1;
    }
}

int get_image_size(image_buffer_t* image)
{
    if (image == NULL) {
        return 0;
    }
    switch (image->format)
    {
    case IMAGE_FORMAT_GRAY8:
        return image->width * image->height;
    case IMAGE_FORMAT_RGB888:
        return image->width * image->height * 3;    
    case IMAGE_FORMAT_RGBA8888:
        return image->width * image->height * 4;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21:
        return image->width * image->height * 3 / 2;
    default:
        break;
    }
}

static int convert_image_rga(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret = 0;

    // ===== 提取源图像信息 =====
    int srcWidth = src_img->width;           // 源图像宽度（像素）
    int srcHeight = src_img->height;         // 源图像高度（像素）
    void *src = src_img->virt_addr;          // 源图像虚拟地址指针
    int src_fd = src_img->fd;                // 源图像文件描述符（用于DMA缓冲区，0表示普通内存）
    void *src_phy = NULL;                    // 源图像物理地址（通常为NULL，由驱动管理）
    int srcFmt = get_rga_fmt(src_img->format); // 将通用格式转换为RGA特定格式枚举

    // ===== 提取目标图像信息 =====
    int dstWidth = dst_img->width;           // 目标图像宽度
    int dstHeight = dst_img->height;         // 目标图像高度
    void *dst = dst_img->virt_addr;          // 目标图像虚拟地址指针
    int dst_fd = dst_img->fd;                // 目标图像文件描述符
    void *dst_phy = NULL;                    // 目标图像物理地址
    int dstFmt = get_rga_fmt(dst_img->format); // 转换为RGA格式枚举

    int rotate = 0;  // 旋转角度标志（0=不旋转，IM_HAL_TRANSFORM_ROT_90等）

    // ===== 检查是否使用RGA handle模式 =====
    // handle模式是新版RGA API，提供更好的内存管理和性能
    // 需要librga支持LIBRGA_IM2D_HANDLE宏定义
    int use_handle = 0;
#if defined(LIBRGA_IM2D_HANDLE)
    use_handle = 1;
#endif

    // RGA操作状态码和使用标志
    int usage = 0;  // RGA操作标志位（旋转、缩放模式等）
    IM_STATUS ret_rga = IM_STATUS_NOERROR;  // RGA操作返回状态

    // ===== 设置RGA使用标志 =====
    usage |= rotate;  // 添加旋转标志到usage（当前rotate=0，即不旋转）

    // ===== 设置RGA处理的矩形区域 =====
    im_rect srect;   // 源图像裁剪矩形（从源图像的哪个区域读取）
    im_rect drect;   // 目标图像放置矩形（写入到目标图像的哪个区域）
    im_rect prect;   // 图案矩形（用于特殊填充模式，当前未使用）
    memset(&prect, 0, sizeof(im_rect));

    // 设置源图像裁剪区域
    if (src_box != NULL) {
        // 使用指定的裁剪区域（left,top是起点，right,bottom是终点）
        srect.x = src_box->left;
        srect.y = src_box->top;
        srect.width = src_box->right - src_box->left + 1;   // +1是因为包含边界点
        srect.height = src_box->bottom - src_box->top + 1;
    } else {
        // 使用整个源图像
        srect.x = 0;
        srect.y = 0;
        srect.width = srcWidth;
        srect.height = srcHeight;
    }

    // 设置目标图像放置区域
    if (dst_box != NULL) {
        // 使用指定的放置区域（源图像经过缩放后放置到这个区域）
        drect.x = dst_box->left;
        drect.y = dst_box->top;
        drect.width = dst_box->right - dst_box->left + 1;
        drect.height = dst_box->bottom - dst_box->top + 1;
    } else {
        // 填充整个目标图像
        drect.x = 0;
        drect.y = 0;
        drect.width = dstWidth;
        drect.height = dstHeight;
    }

    // ===== 设置RGA缓冲区结构 =====
    rga_buffer_t rga_buf_src;               // 源图像RGA缓冲区封装
    rga_buffer_t rga_buf_dst;               // 目标图像RGA缓冲区封装
    rga_buffer_t pat;                       // 图案缓冲区（未使用）
    rga_buffer_handle_t rga_handle_src = 0; // 源图像句柄（handle模式）
    rga_buffer_handle_t rga_handle_dst = 0; // 目标图像句柄（handle模式）
    memset(&pat, 0, sizeof(rga_buffer_t));

    // 源图像参数（用于handle模式的导入）
    im_handle_param_t in_param;
    in_param.width = srcWidth;
    in_param.height = srcHeight;
    in_param.format = srcFmt;

    // 目标图像参数（用于handle模式的导入）
    im_handle_param_t dst_param;
    dst_param.width = dstWidth;
    dst_param.height = dstHeight;
    dst_param.format = dstFmt;

    // ===== 导入源图像缓冲区到RGA =====
    if (use_handle) {
        // 使用handle模式导入（新版API，推荐）
        if (src_phy != NULL) {
            // 从物理地址导入（少见，通常由驱动分配）
            rga_handle_src = importbuffer_physicaladdr((uint64_t)src_phy, &in_param);
        } else if (src_fd > 0) {
            // 从文件描述符导入（DMA缓冲区，性能最佳）
            // 这是最推荐的方式，RGA可以直接访问物理连续内存
            rga_handle_src = importbuffer_fd(src_fd, &in_param);
        } else {
            // 从虚拟地址导入（普通malloc内存）
            // RGA驱动会处理虚拟地址到物理地址的转换，可能有性能损失
            rga_handle_src = importbuffer_virtualaddr(src, &in_param);
        }
        // 检查句柄是否有效
        if (rga_handle_src <= 0) {
            printf("src handle error %d\n", rga_handle_src);
            ret = -1;
            goto err;
        }
        // 将句柄封装为RGA缓冲区对象
        // 参数: handle, 宽, 高, 格式, 虚拟宽度(用于对齐), 虚拟高度
        rga_buf_src = wrapbuffer_handle(rga_handle_src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
    } else {
        // 使用旧版API直接封装缓冲区（不推荐，兼容性用）
        if (src_phy != NULL) {
            rga_buf_src = wrapbuffer_physicaladdr(src_phy, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else if (src_fd > 0) {
            rga_buf_src = wrapbuffer_fd(src_fd, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else {
            rga_buf_src = wrapbuffer_virtualaddr(src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        }
    }

    // ===== 导入目标图像缓冲区到RGA =====
    if (use_handle) {
        // 使用handle模式导入
        if (dst_phy != NULL) {
            rga_handle_dst = importbuffer_physicaladdr((uint64_t)dst_phy, &dst_param);
        } else if (dst_fd > 0) {
            rga_handle_dst = importbuffer_fd(dst_fd, &dst_param);
        } else {
            rga_handle_dst = importbuffer_virtualaddr(dst, &dst_param);
        }
        // 检查句柄是否有效
        if (rga_handle_dst <= 0) {
            printf("dst handle error %d\n", rga_handle_dst);
            ret = -1;
            goto err;
        }
        // 封装为RGA缓冲区对象
        rga_buf_dst = wrapbuffer_handle(rga_handle_dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
    } else {
        // 使用旧版API直接封装
        if (dst_phy != NULL) {
            rga_buf_dst = wrapbuffer_physicaladdr(dst_phy, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else if (dst_fd > 0) {
            rga_buf_dst = wrapbuffer_fd(dst_fd, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else {
            rga_buf_dst = wrapbuffer_virtualaddr(dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        }
    }

    // ===== 填充目标图像背景色（letterbox的关键步骤）=====
    // 当目标放置区域小于整个目标图像时，需要填充背景色
    // 例如: 480x640 -> 640x640，左右各填充80像素的黑边
    if (drect.width != dstWidth || drect.height != dstHeight) {
        im_rect dst_whole_rect = {0, 0, dstWidth, dstHeight};
        
        // 构造RGBA格式的颜色值（所有通道填充相同的color值）
        int imcolor;
        char* p_imcolor = (char*)&imcolor;
        p_imcolor[0] = color;  // R通道
        p_imcolor[1] = color;  // G通道
        p_imcolor[2] = color;  // B通道
        p_imcolor[3] = color;  // A通道
        
        printf("fill dst image (x y w h)=(%d %d %d %d) with color=0x%x\n",
            dst_whole_rect.x, dst_whole_rect.y, dst_whole_rect.width, dst_whole_rect.height, imcolor);
        
        // ⚠️ 使用RGA的imfill函数填充背景色
        // 注意: 此函数在某些情况下可能失败（如内存未对齐、驱动问题等）
        ret_rga = imfill(rga_buf_dst, dst_whole_rect, imcolor);
        if (ret_rga <= 0) {
            // RGA填充失败，降级到CPU填充（使用memset）
            if (dst != NULL) {
                size_t dst_size = get_image_size(dst_img);
                memset(dst, color, dst_size);
            } else {
                printf("Warning: Can not fill color on target image\n");
            }
        }
    }

    // ===== 执行RGA图像处理（核心操作）=====
    // improcess是RGA的核心函数，可以同时完成:
    // 1. 从源图像裁剪 (srect)
    // 2. 缩放到目标尺寸 (drect)
    // 3. 格式转换 (srcFmt -> dstFmt)
    // 4. 旋转/镜像 (usage标志)
    // 参数: 源缓冲区, 目标缓冲区, 图案缓冲区(NULL), 源矩形, 目标矩形, 图案矩形(NULL), 使用标志
    ret_rga = improcess(rga_buf_src, rga_buf_dst, pat, srect, drect, prect, usage);
    if (ret_rga <= 0) {
        // RGA处理失败，打印错误信息
        printf("Error on improcess STATUS=%d\n", ret_rga);
        printf("RGA error message: %s\n", imStrError((IM_STATUS)ret_rga));
        ret = -1;
    }

err:
    // ===== 清理资源 =====
    // 释放源图像句柄（handle模式需要显式释放）
    if (rga_handle_src > 0) {
        releasebuffer_handle(rga_handle_src);
    }

    // 释放目标图像句柄
    if (rga_handle_dst > 0) {
        releasebuffer_handle(rga_handle_dst);
    }

    return ret;
}

/**
 * @brief 图像格式转换和缩放的统一接口
 * 该函数会自动选择最优的处理方式：优先使用RGA硬件加速，失败时降级到CPU处理
 * 
 * @param src_img 源图像缓冲区
 * @param dst_img 目标图像缓冲区
 * @param src_box 源图像的裁剪区域（NULL表示使用整个图像）
 * @param dst_box 目标图像的放置区域（NULL表示填充整个目标图像）
 * @param color 填充颜色值（用于填充目标图像中未被源图像覆盖的区域）
 * @return int 0表示成功，-1表示失败
 */
int convert_image(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret;
 
    // ===== 打印调试信息 =====
    // 输出源图像的详细信息
    printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        src_img->width,        // 源图像宽度
        src_img->height,       // 源图像高度
        src_img->format,       // 源图像格式（如RGB888、NV12等）
        src_img->virt_addr,    // 源图像虚拟地址指针
        src_img->fd);          // 源图像文件描述符（用于DMA缓冲区）
    
    // 输出目标图像的详细信息
    printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        dst_img->width,        // 目标图像宽度
        dst_img->height,       // 目标图像高度
        dst_img->format,       // 目标图像格式
        dst_img->virt_addr,    // 目标图像虚拟地址指针
        dst_img->fd);          // 目标图像文件描述符
    
    // 如果指定了源裁剪区域，输出裁剪框坐标
    if (src_box != NULL) {
        printf("src_box=(%d %d %d %d)\n", 
            src_box->left,     // 裁剪区域左边界
            src_box->top,      // 裁剪区域上边界
            src_box->right,    // 裁剪区域右边界
            src_box->bottom);  // 裁剪区域下边界
    }
    
    // 如果指定了目标放置区域，输出目标框坐标
    if (dst_box != NULL) {
        printf("dst_box=(%d %d %d %d)\n", 
            dst_box->left,     // 目标区域左边界
            dst_box->top,      // 目标区域上边界
            dst_box->right,    // 目标区域右边界
            dst_box->bottom);  // 目标区域下边界
    }
    
    // 输出填充颜色值
    printf("color=0x%x\n", color);

    // ===== 尝试使用RGA硬件加速 =====
    // RGA (Raster Graphic Acceleration) 是Rockchip的2D图形加速器
    // 可以高效处理图像缩放、旋转、格式转换等操作
    ret = convert_image_rga(src_img, dst_img, src_box, dst_box, color);
    
    // ===== 降级到CPU处理 =====
    // 如果RGA处理失败（可能原因：硬件不支持、格式不兼容、驱动问题等）
    // 则使用CPU进行软件处理（速度较慢但兼容性好）
    if (ret != 0) {
        printf("try convert image use cpu\n");
        ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    }
    
    return ret;
}

int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color)
{
    int ret = 0;
    int allow_slight_change = 1;
    int src_w = src_image->width;
    int src_h = src_image->height;
    int dst_w = dst_image->width;
    int dst_h = dst_image->height;
    int resize_w = dst_w;
    int resize_h = dst_h;

    int padding_w = 0;
    int padding_h = 0;

    int _left_offset = 0;
    int _top_offset = 0;
    float scale = 1.0;

    image_rect_t src_box;
    src_box.left = 0;
    src_box.top = 0;
    src_box.right = src_image->width - 1;
    src_box.bottom = src_image->height - 1;

    image_rect_t dst_box;
    dst_box.left = 0;
    dst_box.top = 0;
    dst_box.right = dst_image->width - 1;
    dst_box.bottom = dst_image->height - 1;

    float _scale_w = (float)dst_w / src_w;
    float _scale_h = (float)dst_h / src_h;
    if(_scale_w < _scale_h) {
        scale = _scale_w;
        resize_h = (int) src_h*scale;
    } else {
        scale = _scale_h;
        resize_w = (int) src_w*scale;
    }
    // slight change image size for align
    if (allow_slight_change == 1 && (resize_w % 4 != 0)) {
        resize_w -= resize_w % 4;
    }
    if (allow_slight_change == 1 && (resize_h % 2 != 0)) {
        resize_h -= resize_h % 2;
    }
    // padding
    padding_h = dst_h - resize_h;
    padding_w = dst_w - resize_w;
    // center
    if (_scale_w < _scale_h) {
        dst_box.top = padding_h / 2;
        if (dst_box.top % 2 != 0) {
            dst_box.top -= dst_box.top % 2;
            if (dst_box.top < 0) {
                dst_box.top = 0;
            }
        }
        dst_box.bottom = dst_box.top + resize_h - 1;
        _top_offset = dst_box.top;
    } else {
        dst_box.left = padding_w / 2;
        if (dst_box.left % 2 != 0) {
            dst_box.left -= dst_box.left % 2;
            if (dst_box.left < 0) {
                dst_box.left = 0;
            }
        }
        dst_box.right = dst_box.left + resize_w - 1;
        _left_offset = dst_box.left;
    }
    printf("scale=%f dst_box=(%d %d %d %d) allow_slight_change=%d _left_offset=%d _top_offset=%d padding_w=%d padding_h=%d\n",
        scale, dst_box.left, dst_box.top, dst_box.right, dst_box.bottom, allow_slight_change,
        _left_offset, _top_offset, padding_w, padding_h);

    //set offset and scale
    if(letterbox != NULL){
        letterbox->scale = scale;
        letterbox->x_pad = _left_offset;
        letterbox->y_pad = _top_offset;
    }
    // alloc memory buffer for dst image,
    // remember to free
    if (dst_image->virt_addr == NULL && dst_image->fd <= 0) {
        int dst_size = get_image_size(dst_image);
        dst_image->virt_addr = (uint8_t *)malloc(dst_size);
        if (dst_image->virt_addr == NULL) {
            printf("malloc size %d error\n", dst_size);
            return -1;
        }
    }
    ret = convert_image(src_image, dst_image, &src_box, &dst_box, color);
    return ret;
}