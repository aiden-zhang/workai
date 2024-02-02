#include <stdio.h>
#include <assert.h>
#include "yuv2yuv_cvt.h"


void YUV444pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream) {
    if (false && w % 16 == 0 && align_w % 16 == 0 && (uint64_t)sy % 16 == 0 && (uint64_t)su % 16 == 0 && (uint64_t)sv % 16 == 0 &&
        (uint64_t)dy % 16 == 0 && (uint64_t)du % 8 == 0 && (uint64_t)dv % 8 == 0)
    {
        int w_new = w / sizeof(ulonglong2);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<ulonglong2, uint64_t, 8><<<grid, block, 0, stream>>>(
            (ulonglong2 *)sy, (ulonglong2 *)su, (ulonglong2 *)sv, (ulonglong2 *)dy, (uint64_t *)du, (uint64_t *)dv,
            w_new, h / 2, align_w / sizeof(ulonglong2));
    }
    else if (false && w % 8 == 0 && align_w % 8 == 0 && (uint64_t)sy % 8 == 0 && (uint64_t)su % 8 == 0 && (uint64_t)sv % 8 == 0 &&
             (uint64_t)dy % 8 == 0 && (uint64_t)du % 4 == 0 && (uint64_t)dv % 4 == 0)
    {
        int w_new = w / sizeof(uint64_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint64_t, uint32_t, 4><<<grid, block, 0, stream>>>(
            (uint64_t *)sy, (uint64_t *)su, (uint64_t *)sv, (uint64_t *)dy, (uint32_t *)du, (uint32_t *)dv,
            w_new, h / 2, align_w / sizeof(uint64_t));
    }
    else if (false && w % 4 == 0 && align_w % 4 == 0 && (uint64_t)sy % 4 == 0 && (uint64_t)su % 4 == 0 && (uint64_t)sv % 4 == 0 &&
             (uint64_t)dy % 4 == 0 && (uint64_t)du % 2 == 0 && (uint64_t)dv % 2 == 0)
    {
        int w_new = w / sizeof(uint32_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint32_t, uint16_t, 2><<<grid, block, 0, stream>>>(
            (uint32_t *)sy, (uint32_t *)su, (uint32_t *)sv, (uint32_t *)dy, (uint16_t *)du, (uint16_t *)dv,
            w_new, h / 2, align_w / sizeof(uint32_t));
    }
    else if (false && w % 2 == 0 && align_w % 2 == 0 && (uint64_t)sy % 2 == 0 && (uint64_t)su % 2 == 0 && (uint64_t)sv % 2 == 0 &&
             (uint64_t)dy % 2 == 0)
    {
        int w_new = w / sizeof(uint16_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<uint16_t, uint8_t, 1><<<grid, block, 0, stream>>>(
            (uint16_t *)sy, (uint16_t *)su, (uint16_t *)sv, (uint16_t *)dy, (uint8_t *)du, (uint8_t *)dv,
            w_new, h / 2, align_w / sizeof(uint16_t));
    }
    else if (w % 2 == 0)
    {
        dim3 block(16, 16, 1);
        dim3 grid((w / 2 + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv444p_to_yuv420p<<<grid, block, 0, stream>>>(sy, su, sv, dy, du, dv, w, h, align_w);
    }
    else
    {
        printf("w is not multiple of 2!\n");
        assert(false);
    }
}

void YUV440pToYUV420p(uint8_t *sy, uint8_t *su, uint8_t *sv, uint8_t *dy,
    uint8_t *du, uint8_t *dv, int w, int h, int align_w, cudaStream_t stream) {
    if (false && w % 16 == 0 && align_w % 16 == 0 && (uint64_t)sy % 16 == 0 && (uint64_t)su % 16 == 0 && (uint64_t)sv % 16 == 0 &&
        (uint64_t)dy % 16 == 0 && (uint64_t)du % 8 == 0 && (uint64_t)dv % 8 == 0)
    {
        int w_new = w / sizeof(ulonglong2);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<ulonglong2, uint64_t, 8><<<grid, block, 0, stream>>>((ulonglong2*)sy, (ulonglong2*)su, (ulonglong2*)sv,
            (ulonglong2*)dy, (uint64_t*)du, (uint64_t*)dv, w_new, h / 2, align_w / sizeof(ulonglong2));
    }
    else if (false && w % 8 == 0 && align_w % 8 == 0 && (uint64_t)sy % 8 == 0 && (uint64_t)su % 8 == 0 && (uint64_t)sv % 8 == 0 &&
             (uint64_t)dy % 8 == 0 && (uint64_t)du % 4 == 0 && (uint64_t)dv % 4 == 0)
    {
        int w_new = w / sizeof(uint64_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint64_t, uint32_t, 4><<<grid, block, 0, stream>>>((uint64_t*)sy, (uint64_t*)su, (uint64_t*)sv,
            (uint64_t*)dy, (uint32_t*)du, (uint32_t*)dv, w_new, h / 2, align_w / sizeof(uint64_t));
    }
    else if (false && w % 4 == 0 && align_w % 4 == 0 && (uint64_t)sy % 4 == 0 && (uint64_t)su % 4 == 0 && (uint64_t)sv % 4 == 0 &&
             (uint64_t)dy % 4 == 0 && (uint64_t)du % 2 == 0 && (uint64_t)dv % 2 == 0)
    {
        int w_new = w / sizeof(uint32_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint32_t, uint16_t, 2><<<grid, block, 0, stream>>>((uint32_t*)sy, (uint32_t*)su, (uint32_t*)sv,
            (uint32_t*)dy, (uint16_t*)du, (uint16_t*)dv, w_new, h / 2, align_w / sizeof(uint32_t));
    }
    else if (false && w % 2 == 0 && align_w % 2 == 0 && (uint64_t)sy % 2 == 0 && (uint64_t)su % 2 == 0 && (uint64_t)sv % 2 == 0 &&
             (uint64_t)dy % 2 == 0)
    {
        int w_new = w / sizeof(uint16_t);
        dim3 block(16, 16, 1);
        dim3 grid((w_new + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<uint16_t, uint8_t, 1><<<grid, block, 0, stream>>>((uint16_t*)sy, (uint16_t*)su, (uint16_t*)sv,
            (uint16_t*)dy, (uint8_t*)du, (uint8_t*)dv, w_new, h / 2, align_w / sizeof(uint16_t));
    }
    else if (w % 2 == 0)
    {
        dim3 block(16, 16, 1);
        dim3 grid((w / 2 + block.x - 1) / block.x, (h / 2 + block.y - 1) / block.y, 1);
        yuv440p_to_yuv420p<<<grid, block, 0, stream>>>(sy, su, sv, dy, du, dv, w, h, align_w);
    }
    else
    {
        printf("w is not multiple of 2!\n");
        assert(false);
    }
}
