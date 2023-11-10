#ifndef IMAGE_PROC_H_
#define IMAGE_PROC_H_

/// @brief Resize RGB image with bilinear interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void RGBResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief Resize RGB image with nearest neighbor interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void RGBResizeNearest(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief Resize gray image with bilinear interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void GrayResizeBilinear(uint8_t *in_buf, uint8_t *out_buf,
                       int w_in, int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief Resize gray image with nearest neighbor interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void GrayResizeNearest(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(nv12) image with nearest neighbor interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVNv12ResizeNearest(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(nv21) image with nearest neighbor interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVNv21ResizeNearest(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(i420) image with nearest neighbor interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVI420ResizeNearest(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(nv12) image with bilinear interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVNv12ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(nv21) image with bilinear interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVNv21ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Resize YUV(i420) image with bilinear interpolation.
///
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3 / 2) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param stream CU kernel run in the stream.
void YUVI420ResizeBilinear(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, int out_w, int out_h, cudaStream_t stream);

/// @brief Convert YUV(yu12, also called i420) image to RGB image.
///        The order of RGB image is RGBRGBRGB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVYu12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Convert YUV(nv12) image to RGB image.
///        The order of RGB image is RGBRGBRGB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVNv12ToRGB(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Convert YUV(yu12, also called i420) image to RGB image.
///        The order of RGB image is RGBRGBRGB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVYu12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Convert YUV(nv12) image to RGB image.
///        The order of RGB image is RGBRGBRGB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVNv12ToRGBFloat(uint8_t* in_buf, float* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Convert YUV(yu12, also called i420) image to RGB image.
///        The order of RGB image is RRRGGGBBB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVYu12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Convert YUV(nv12) image to RGB image.
///        The order of RGB image is RRRGGGBBB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param stream CU kernel run in the stream.
void YUVNv12ToRGBPlane(uint8_t* in_buf, uint8_t* out_buf,
                       int in_w, int in_h, cudaStream_t stream);

/// @brief Resize RGB image ROI area with nearset interpolation.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param stream CU kernel run in the stream.
void RGBResizeWithROINearest(uint8_t *in_buf, uint8_t *out_buf,
                            int w_in, int h_in, int w_out, int h_out,
                            int roi_w_start, int roi_h_start, int roi_w, 
                            int roi_h, cudaStream_t stream);

/// @brief Resize RGB image ROI area with bilinear interpolation.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param stream CU kernel run in the stream.
void RGBResizeWithROIBilinear(uint8_t *in_buf, uint8_t *out_buf,
                            int w_in, int h_in, int w_out, int h_out,
                            int roi_w_start, int roi_h_start, int roi_w, 
                            int roi_h, cudaStream_t stream);

/// @brief Resize RGB image with bilinear interpolation.
///        The order of RGB image is RRRGGGBBB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void RGBResizePlaneBilinear(uint8_t *in_buf, uint8_t *out_buf,
                            int w_in, int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief Resize RGB image with nearest interpolation.
///        The order of RGB image is RRRGGGBBB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void RGBResizePlaneNearest(uint8_t *in_buf, uint8_t *out_buf,
                            int w_in, int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief Resize RGB image with bilinear interpolation.
///        The order of input and output RGB image is RRRGGGBBB.
///        Output image has padding.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_box * h_box * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of resized image.
/// @param h_out The height of resized image.
/// @param w_box The width of output image.
/// @param h_box The height of output image.
/// @param w_b The offset of width dimension of resized image in output image.
/// @param h_b The offset of height dimension of resized image in output image.
/// @param stream CU kernel run in the stream.
void RGBResizePlanePadBilinear(uint8_t *in_buf, uint8_t *out_buf,
                              int w_in, int h_in, int w_out, int h_out, 
                              int w_box, int h_box, int w_b, int h_b, cudaStream_t stream);

/// @brief Resize RGB image with nearest interpolation.
///        The order of input and output RGB image is RRRGGGBBB.
///        Output image has padding.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_box * h_box * 3) * sizeof(uint8_t).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of resized image.
/// @param h_out The height of resized image.
/// @param w_box The width of output image.
/// @param h_box The height of output image.
/// @param w_b The offset of width dimension of resized image in output image.
/// @param h_b The offset of height dimension of resized image in output image.
/// @param stream CU kernel run in the stream.
void RGBResizePlanePadNearest(uint8_t *in_buf, uint8_t *out_buf,
                            int w_in, int h_in, int w_out, int h_out, 
                            int w_box, int h_box, int w_b, int h_b, cudaStream_t stream);

/// @brief Crop RGB image.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(uint8_t).
/// @param start_w The start index of crop area in the width dimension of intput image.
/// @param start_h The start index of crop area in the height dimension of intput image.
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param stream CU kernel run in the stream.
void RGBCrop(uint8_t *in_buf, uint8_t *out_buf,
            int start_w, int start_h, int w_in, 
            int h_in, int w_out, int h_out, cudaStream_t stream);

/// @brief RGB image normalization.
///        output = (input * scale - mean) * standard
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * in_c) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * in_c) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param in_c The channels of input image.
/// @param mean, standard, scale The parameter in "output = (input * scale - mean) * standard"
/// @param stream CU kernel run in the stream.
void RGBNormalization(uint8_t* in_buf, float* out_buf,
                      int in_w, int in_h, int in_c, float mean, 
                      float standard, float scale, cudaStream_t stream);

/// @brief RGB image normalization.
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param mean1, mean2, mean3, standard1, standard2, standard3, scale
///        The parameter in "output[channel_out] = (input[channel] * scale - mean[channel]) * standard[channel]".
/// @param input_plane True mean input image is RRRGGGBBB.
///        False mean input image is RGBRGBRGB.
/// @param output_plane True mean output image is RRRGGGBBB.
///        False mean output image is RGBRGBRGB.
/// @param channel_rev True mean rgb convert to bgr or bgr convert to rgb.
///        False mean input and output image is same format.
/// @param stream CU kernel run in the stream.
void RGBNormalization_3Channels(uint8_t* in_buf, float* out_buf,
                                int in_w, int in_h, float mean1, float mean2, float mean3,
                                float standard1, float standard2, float standard3, float scale, 
                                bool input_plane, bool output_plane, bool channel_rev, cudaStream_t stream);

/// @brief NV12 image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void NV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void YU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void NV12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void YU12ToRGBNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 image convert to BGR and resize with padding and normalization.
///        The order of BGR image is BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void NV12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) image convert to BGR and resize with padding and normalization.
///        The order of BGR image is BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void YU12ToBGRBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 image convert to BGR and resize with padding and normalization.
///        The order of BGR image is BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void NV12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) image convert to BGR and resize with padding and normalization.
///        The order of BGR image is BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void YU12ToBGRNearestResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief Resize RGB/BGR image ROI area with nearset interpolation and normalization.
///        Output format is RRRGGGBBB / BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(float).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param channel_rev True mean rgb convert to bgr or bgr convert to rgb.
///        False mean input and output image is same format.
/// @param stream CU kernel run in the stream.
void RGBROINearestResizeNormPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, bool channel_rev,
        cudaStream_t stream);

/// @brief Resize RGB/BGR image ROI area with bilinear interpolation and normalization.
///        Output format is RRRGGGBBB / BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(float).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param channel_rev True mean rgb convert to bgr or bgr convert to rgb.
///        False mean input and output image is same format.
/// @param stream CU kernel run in the stream.
void RGBROIBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, bool channel_rev,
        cudaStream_t stream);

/// @brief NV12 image roi area convert to RGB and resize with padding.
///        The order of RGB image is RRRGGGBBB.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiNV12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream);
/// @brief YU12(i420) image roi area convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU12ToRGBBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w,
    int out_h, int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 image roi area convert to BGR and resize with padding.
///        The order of BGR image is BBBGGGRRR.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiNV12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream);
/// @brief YU12(i420) image roi area convert to BGR and resize with padding and normalization.
///        The order of BGR image is BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU12ToBGRBilinearResizePlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w,
    int out_h, int pad_w, int pad_h, float pad1, float pad2, float pad3, cudaStream_t stream);
/// @brief NV12 image roi area convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) image roi area convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 full range(0~255) image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void FullRangeNV12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12(i420) full range(0~255) image convert to RGB and resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void FullRangeYU12ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);


/// @brief Affine transformation gray image with bilinear interpolation.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param stream CU kernel run in the stream.
void GrayAffine(uint8_t *input, uint8_t *output, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h,
    float m[6], cudaStream_t stream);

/// @brief Affine transformation rgb image with bilinear interpolation.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param stream CU kernel run in the stream.
void RGBAffine(uint8_t* input, uint8_t* output, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h,
    int out_w, int out_h, float m[6], cudaStream_t stream);

/// @brief YU12(i420) image convert to RGB and affine transformation.
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(uint8_t).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param stream CU kernel run in the stream.
void YU122RGBAffine(uint8_t* input, uint8_t* output,
    int in_w, int in_h, int out_w, int out_h, float m[6], cudaStream_t stream);

/// @brief YUV444P image roi area convert to RGB and bilinear resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYUV444PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YUV400P image roi area convert to RGB and bilinear resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean, std, scale The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYUV400PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean, float std, float scale, float pad, cudaStream_t stream);

/// @brief YUV422P image roi area convert to RGB and bilinear resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYUV422PToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YUV422 image roi area convert to RGB and bilinear resize with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYUV422ToRGBBilinearResizeNormPlane(uint8_t *in_buf, float *out_buf, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, float pad1, float pad2, float pad3, cudaStream_t stream);
/// @brief NV12 image roi area convert to RGB and affine transformation with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiNv122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12 image roi area convert to RGB and affine transformation with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU122RGBAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief YU12 image roi area convert to BGR and affine transformation with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param m[6] Affine transformation matrix.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU122BGRAffineNorm(uint8_t* input, float* output,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        int out_w, int out_h, float m[6], float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float scale,
        float pad1, float pad2, float pad3, cudaStream_t stream);

/// @brief NV12 image roi area convert to RGB, bilinear resize, padding and
///        quantize output to uint8.
///        The order of RGB image is RRRGGGBBB.
///        Quantize function is:
///        output[channel] = (input * scales_input + zero_point
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3) * sizeof(uint8_t).
/// @param ws The workspace buffer allocate in device memory.
///               The size is (roi_w * roi_h) * sizeof(uchar4).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param zero_point The output quantize parameter.
/// @param scales_input The output quantize parameter.
/// @param stream CU kernel run in the stream.
void RoiNV12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream);

/// @brief YU12 image roi area convert to RGB, bilinear resize, padding and
///        quantize output to uint8.
///        The order of RGB image is RRRGGGBBB.
///        Quantize function is:
///        output[channel] = (input * scales_input + zero_point
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_h * 3) * sizeof(uint8_t).
/// @param ws The workspace buffer allocate in device memory.
///               The size is (roi_w * roi_h) * sizeof(uchar4).
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param pad1, pad2, pad3 The padding value of output.
/// @param zero_point The output quantize parameter.
/// @param scales_input The output quantize parameter.
/// @param stream CU kernel run in the stream.
void RoiYU12ToRGBBilinearResizeQuantizePlane(uint8_t *in_buf, uint8_t *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h,
    int pad_w, int pad_h, float pad1, float pad2, float pad3, float zero_point, float scales_input, cudaStream_t stream);


/// @brief YU12(i420) image roi area convert to RGB and resize(compatible with transforms.Resize of torchvison) with padding and normalization.
///        The order of RGB image is RRRGGGBBB.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3 / 2) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3) * sizeof(float).
/// @param ws The workspace buffer allocate in device memory.
/// @param in_w The width of input image.
/// @param in_h The height of input image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param out_w The width of output image.
/// @param out_h The height of output image.
/// @param input_pad_w The pad of width dimension of input image.
/// @param input_pad_h The pad of height dimension of input image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param stream CU kernel run in the stream.
void RoiYU12ToRGBBilinearResizeNormPlaneV2(uint8_t *in_buf, float *out_buf, uchar4 *ws, int in_w, int in_h,
    int roi_w_start, int roi_h_start, int roi_w, int roi_h, int out_w, int out_h, int input_pad_w, int input_pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3,
    float scale, uint8_t pad1, uint8_t pad2, uint8_t pad3, cudaStream_t stream);

/// @brief RGB image convert to YUV420p(I420).
/// @param in_buf The input buffer allocate in device memory.
///               The size is (in_w * in_h * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (out_w * out_w * 3 / 2) * sizeof(uint8_t).
/// @param w The width of input image.
/// @param h The height of input image.
/// @param stream CU kernel run in the stream.
void RGB2YU12(uint8_t *in_buf, uint8_t *out_buf, int w, int h, cudaStream_t stream);


/// @brief Sort data with radix sort algorithm.
/// @param data_dev  The keys for sort.
/// @param indxs_dev The indexes which pair with key.
///                  Calculate by call this function with nullptr of data_dev, indxs_dev, data_out_dev and indxs_out_dev.
/// @param total_num The total number of keys need to be sort.
/// @param data_out_dev The sorted keys.
/// @param data_out_dev The sorted indexes.
/// @param workspace_dev The workspace buffer allocate in device memory.
/// @param stream CU kernel run in the stream.
template<typename DataType,typename IndxType,bool IsAscending=false>
int32_t RadixSortFunc(DataType* data_dev,IndxType* indxs_dev,int32_t total_num,
                      DataType* data_out_dev=nullptr,
                      IndxType* indxs_out_dev=nullptr,
                      void* workspace_dev=nullptr,
                      cudaStream_t* stream=nullptr);

/// @brief Non max suppression inner class.
///        The num_bboxes need has same value between batches.
///        At each one batch, the predict data and out data:
///        predict size:         num_bboxes * (5 + num_classes);
///        predict item:         [[cx, cy, w, h, obj_conf, cls0_conf, ..., clsN_conf] ... [cx, cy, w, h, obj_conf, cls0_conf, ..., clsN_conf]];
///        pout size:            1 + max_objects * 7
///        pout item:            [box_cnt, [x1, y1, x2, y2, confidence, class_id, keep_flag], ... [x1, y1, x2, y2, confidence, keep_flag]]
/// @param predict               The multi-batches input buffer allocate in device memory.
/// @param num_batch             The number of batch.
/// @param num_classes           The number of class.
/// @param confidence_threshold  The confidence threshold used for filter out boxes.
/// @param nms_threshold         The nms threshold used for merge boxes.
/// @param pout                  The multi-batches output buffer allocate in device memory.
/// @param max_objects           The max boxes to keep
/// @param stream                The CU kernel run in the stream.
void non_max_suppression(float* predict, int num_batch, int num_bboxes, int num_classes,
                         float confidence_threshold, float nms_threshold, float* pout,
                         int max_objects, cudaStream_t stream);

/// @brief yuv444p transform to yuv420p.
/// @param sy, su, sv The input buffer of [Y, U, V] 3 channels allocate in device memory.
/// @param dy, du, dv The output buffer of [Y, U, V] 3 channels allocate in device memory.
/// @param w The width of image.
/// @param h The height of image.
/// @param align_w The stride of height of image.
/// @param stream CU kernel run in the stream.
void YUV444pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream);

/// @brief yuv440p transform to yuv420p.
/// @param sy, su, sv The input buffer of [Y, U, V] 3 channels allocate in device memory.
/// @param dy, du, dv The output buffer of [Y, U, V] 3 channels allocate in device memory.
/// @param w The width of image.
/// @param h The height of image.
/// @param align_w The stride of height of image.
/// @param stream CU kernel run in the stream.
void YUV440pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream);

/// @brief Resize RGB/BGR image ROI area with nearset interpolation and normalization.
///        Output format is RRRGGGBBB / BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(float).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param channel_rev True mean rgb convert to bgr or bgr convert to rgb.
///        False mean input and output image is same format.
/// @param stream CU kernel run in the stream.
void RGBROINearestResizeNormPadPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out, int img_w, int img_h, int pad_w, int pad_h,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float pad1, float pad2, float pad3, bool channel_rev,
        cudaStream_t stream);

/// @brief Resize RGB/BGR image ROI area with bilinear interpolation and normalization.
///        Output format is RRRGGGBBB / BBBGGGRRR.
///        Normalization function is:
///        output[channel] = (input[channel] * scale - mean[channel]) * standard[channel]
/// @param in_buf The input buffer allocate in device memory.
///               The size is (w_in * h_in * 3) * sizeof(uint8_t).
/// @param out_buf The output buffer allocate in device memory.
///               The size is (w_out * h_out * 3) * sizeof(float).
/// @param w_in The width of input image.
/// @param h_in The height of input image.
/// @param w_out The width of output image.
/// @param h_out The height of output image.
/// @param img_w The width of resized image.
/// @param img_h The height of resized image.
/// @param pad_w The offset of width dimension of resized image in output image.
/// @param pad_h The offset of height dimension of resized image in output image.
/// @param roi_w_start The start index of ROI area in the width dimension of intput image.
/// @param roi_h_start The start index of ROI area in the height dimension of intput image.
/// @param roi_w The width of ROI area in the width dimension of intput image.
/// @param roi_h The height of ROI area in the height dimension of intput image.
/// @param mean1, mean2, mean3, std1, std2, std3, scale
///        The parameter in "output[channel] = (input[channel] * scale - mean[channel]) * std[channel]".
/// @param pad1, pad2, pad3 The padding value of output.
/// @param channel_rev True mean rgb convert to bgr or bgr convert to rgb.
///        False mean input and output image is same format.
/// @param stream CU kernel run in the stream.
void RGBROIBilinearResizeNormPadPlane(uint8_t *in_buf, float *out_buf,
        int w_in, int h_in, int w_out, int h_out, int img_w, int img_h, int pad_w, int pad_h,
        int roi_w_start, int roi_h_start, int roi_w, int roi_h,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, float pad1, float pad2, float pad3, bool channel_rev,
        cudaStream_t stream);
#endif /* IMAGE_PROC_H_ */
