#include <stdio.h>
#include <assert.h>
#include "yuv2yuv_cvt.h"


void YUV444pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream) {
    if (w % 16 == 0 && align_w % 16 == 0) {
        int w_new = w / sizeof(ulonglong2);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<ulonglong2, uint64_t, 8><<<grid, block, 0, stream>>>((ulonglong2*)sy, (ulonglong2*)su, (ulonglong2*)sv,
            (ulonglong2*)dy, (uint64_t*)du, (uint64_t*)dv, w_new, h / 2, align_w / sizeof(ulonglong2)); 
    } else if (w % 8 == 0 && align_w % 8 == 0) {
        int w_new = w / sizeof(uint64_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint64_t, uint32_t, 4><<<grid, block, 0, stream>>>((uint64_t*)sy, (uint64_t*)su, (uint64_t*)sv,
            (uint64_t*)dy, (uint32_t*)du, (uint32_t*)dv, w_new, h / 2, align_w / sizeof(uint64_t)); 
    } else if (w % 4 == 0 && align_w % 4 == 0) {
        int w_new = w / sizeof(uint32_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint32_t, uint16_t, 2><<<grid, block, 0, stream>>>((uint32_t*)sy, (uint32_t*)su, (uint32_t*)sv,
            (uint32_t*)dy, (uint16_t*)du, (uint16_t*)dv, w_new, h / 2, align_w / sizeof(uint32_t)); 
    } else if (w % 2 == 0 && align_w % 2 == 0) {
        int w_new = w / sizeof(uint16_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint16_t, uint8_t, 1><<<grid, block, 0, stream>>>((uint16_t*)sy, (uint16_t*)su, (uint16_t*)sv,
            (uint16_t*)dy, (uint8_t*)du, (uint8_t*)dv, w_new, h / 2, align_w / sizeof(uint16_t)); 
    } else {
        printf("w or align_w is not multiple of 2!\n");
        assert(false);
    }
}

void YUV440pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream) {
    if (w % 16 == 0 && align_w % 16 == 0) {
        int w_new = w / sizeof(ulonglong2);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<ulonglong2, uint64_t, 8><<<grid, block, 0, stream>>>((ulonglong2*)sy, (ulonglong2*)su, (ulonglong2*)sv,
            (ulonglong2*)dy, (uint64_t*)du, (uint64_t*)dv, w_new, h / 2, align_w / sizeof(ulonglong2)); 
    } else if (w % 8 == 0 && align_w % 8 == 0) {
        int w_new = w / sizeof(uint64_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint64_t, uint32_t, 4><<<grid, block, 0, stream>>>((uint64_t*)sy, (uint64_t*)su, (uint64_t*)sv,
            (uint64_t*)dy, (uint32_t*)du, (uint32_t*)dv, w_new, h / 2, align_w / sizeof(uint64_t)); 
    } else if (w % 4 == 0 && align_w % 4 == 0) {
        int w_new = w / sizeof(uint32_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint32_t, uint16_t, 2><<<grid, block, 0, stream>>>((uint32_t*)sy, (uint32_t*)su, (uint32_t*)sv,
            (uint32_t*)dy, (uint16_t*)du, (uint16_t*)dv, w_new, h / 2, align_w / sizeof(uint32_t)); 
    } else if (w % 2 == 0 && align_w % 2 == 0) {
        int w_new = w / sizeof(uint16_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint16_t, uint8_t, 1><<<grid, block, 0, stream>>>((uint16_t*)sy, (uint16_t*)su, (uint16_t*)sv,
            (uint16_t*)dy, (uint8_t*)du, (uint8_t*)dv, w_new, h / 2, align_w / sizeof(uint16_t)); 
    } else {
        printf("w or align_w is not multiple of 2!\n");
        assert(false);
    }
}
