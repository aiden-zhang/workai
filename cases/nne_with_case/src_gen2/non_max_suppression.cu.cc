#include <iostream>
#include "common.h"
#include "non_max_suppression.h"

void non_max_suppression(
    float* predict, int num_batch, int num_bboxes, int num_classes, float confidence_threshold,
    float nms_threshold, float* pout, int max_objects, cudaStream_t stream){

    int num_box_element = 7;

    int per_tile_bboxes = ceil(num_bboxes / float(NUM_TILE));
    int per_tile_out_size = max_objects * num_box_element;
    float* filter_ws;
    int *nms_ws;
    CUDA_CHECK(cudaMalloc(&filter_ws, NUM_TILE * per_tile_out_size * sizeof(float) + NUM_TILE * sizeof(float)));

    dim3 block_nms(128, 1, 1);
    dim3 grid_nms((max_objects + block_nms.x - 1) / block_nms.x ,1,1);
    CUDA_CHECK(cudaMalloc(&nms_ws, grid_nms.x * (num_classes + block_nms.x) * sizeof(int)));

    dim3 block_flt(512,1,1);
    dim3 grid_flt(NUM_TILE,1,1);

    for (int ib=0; ib<num_batch; ib++) {
        float *predict_batch = predict + ib * num_bboxes * (num_classes + 5);
        float *pout_batch = pout + ib * (max_objects * num_box_element + 1);

        filter_kernel<<<grid_flt, block_flt, 0, stream>>>(
                predict_batch, filter_ws,
                num_bboxes, num_classes, NUM_TILE,
                confidence_threshold, max_objects,
                num_box_element, per_tile_bboxes);

        dim3 block_mtk(128,1,1);
        dim3 grid_mtk(NUM_TILE,1,1);
        int sm_size = NUM_TILE * 2 * sizeof(int);
        merge_tile_kernel<<<grid_mtk, block_mtk, sm_size, stream>>>(filter_ws, pout_batch, per_tile_out_size, max_objects, num_box_element);

#if 0
        float fcnt = 0.0f;
        cudaMemcpyAsync(&fcnt, pout_batch, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        printf("out bbox count: %.2f\n", fcnt);
#endif
        classify<<<grid_nms, block_nms, (num_classes + block_nms.x * num_classes) * sizeof(int), stream>>>(pout_batch, nms_ws, max_objects, num_classes, num_box_element);
        fast_nms_kernel_opt<<<grid_nms, block_nms, 0, stream>>>(pout_batch, nms_ws, max_objects, num_classes, nms_threshold, num_box_element);
    }
    cudaStreamSynchronize(stream);

    cudaFree(filter_ws);
    cudaFree(nms_ws);
}
