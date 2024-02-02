#include <stdio.h>
#include "yuv_crop.h"

void YU12Crop(uint8_t *in_buf, uint8_t *out_buf,
              int start_w, int start_h, int w_in,
              int h_in, int w_out, int h_out, cudaStream_t stream)
{
    if (start_w % 2 != 0 || start_h % 2 != 0)
    {
        printf("error:func 'YU12Crop' start_w or start_h  not align to 2 !\n");
    }
    uint8_t *in_y = in_buf;
    uint8_t *in_u = in_y + w_in * h_in;
    uint8_t *in_v = in_u + (w_in * h_in >> 2);
    uint8_t *out_y = out_buf;
    uint8_t *out_u = out_y + w_out * h_out;
    uint8_t *out_v = out_u + (w_out * h_out >> 2);
    if (w_in % 8 == 0 && start_w % 8 == 0 && w_out % 8 == 0 && (uint64_t)in_buf % 8 == 0 && (uint64_t)out_buf % 8 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 8 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        yu12_crop_kernel<uint64_t, uint32_t><<<grid, block, 0, stream>>>((uint64_t *)in_y, (uint32_t *)in_u, (uint32_t *)in_v,
                                                                         (uint64_t *)out_y, (uint32_t *)out_u, (uint32_t *)out_v,
                                                                         start_h, start_w / 8, h_in, w_in / 8, h_out, w_out / 8);
    }
    else if (w_in % 4 == 0 && start_w % 4 == 0 && w_out % 4 == 0 && (uint64_t)in_buf % 4 == 0 && (uint64_t)out_buf % 4 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 4 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        yu12_crop_kernel<uint32_t, uint16_t><<<grid, block, 0, stream>>>((uint32_t *)in_y, (uint16_t *)in_u, (uint16_t *)in_v,
                                                                         (uint32_t *)out_y, (uint16_t *)out_u, (uint16_t *)out_v,
                                                                         start_h, start_w / 4, h_in, w_in / 4, h_out, w_out / 4);
    }
    else if (w_in % 2 == 0 && start_w % 2 == 0 && w_out % 2 == 0 && (uint64_t)in_buf % 2 == 0 && (uint64_t)out_buf % 2 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 2 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        yu12_crop_kernel<uint16_t, uint8_t><<<grid, block, 0, stream>>>((uint16_t *)in_y, (uint8_t *)in_u, (uint8_t *)in_v,
                                                                        (uint16_t *)out_y, (uint8_t *)out_u, (uint8_t *)out_v,
                                                                        start_h, start_w / 2, h_in, w_in / 2, h_out, w_out / 2);
    }
    else
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        yu12_crop_kernel<<<grid, block, 0, stream>>>(in_y, in_u, in_v, out_y, out_u, out_v,
                                                     start_h, start_w, h_in, w_in, h_out, w_out);
    }
}

void NV12Crop(uint8_t *in_buf, uint8_t *out_buf,
              int start_w, int start_h, int w_in,
              int h_in, int w_out, int h_out, cudaStream_t stream)
{
    if (start_w % 2 != 0 || start_h % 2 != 0)
    {
        printf("error:func 'NV12Crop' start_w or start_h  not align to 2 !\n");
    }
    uint8_t *in_y = in_buf;
    uint8_t *in_uv = in_y + w_in * h_in;
    uint8_t *out_y = out_buf;
    uint8_t *out_uv = out_y + w_out * h_out;
    if (w_in % 8 == 0 && start_w % 8 == 0 && w_out % 8 == 0 && (uint64_t)in_buf % 8 == 0 && (uint64_t)out_buf % 8 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 8 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        nv12_crop_kernel<uint64_t><<<grid, block, 0, stream>>>((uint64_t *)in_y, (uint64_t *)in_uv, (uint64_t *)out_y, (uint64_t *)out_uv,
                                                               start_h, start_w / 8, h_in, w_in / 8, h_out, w_out / 8);
    }
    else if (w_in % 4 == 0 && start_w % 4 == 0 && w_out % 4 == 0 && (uint64_t)in_buf % 4 == 0 && (uint64_t)out_buf % 4 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 4 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        nv12_crop_kernel<uint32_t><<<grid, block, 0, stream>>>((uint32_t *)in_y, (uint32_t *)in_uv, (uint32_t *)out_y, (uint32_t *)out_uv,
                                                               start_h, start_w / 4, h_in, w_in / 4, h_out, w_out / 4);
    }
    else if (w_in % 2 == 0 && start_w % 2 == 0 && w_out % 2 == 0 && (uint64_t)in_buf % 2 == 0 && (uint64_t)out_buf % 2 == 0)
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out / 2 + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        nv12_crop_kernel<uint16_t><<<grid, block, 0, stream>>>((uint16_t *)in_y, (uint16_t *)in_uv, (uint16_t *)out_y, (uint16_t *)out_uv,
                                                               start_h, start_w / 2, h_in, w_in / 2, h_out, w_out / 2);
    }
    else
    {
        dim3 block(32, 4, 1);
        dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1) / block.y, 1);
        nv12_crop_kernel<<<grid, block, 0, stream>>>(in_y, in_uv, out_y, out_uv,
                                                     start_h, start_w, h_in, w_in, h_out, w_out);
    }
}
