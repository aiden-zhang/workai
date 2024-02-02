#ifndef NON_MAX_SUPPRESSION_H_
#define NON_MAX_SUPPRESSION_H_

#define NUM_TILE  32

static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom,
        float bleft, float btop, float bright, float bbottom
        ){

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void classify(
        float * __restrict__ bboxes, int * __restrict__ ws, int max_objects, int num_classes, int NUM_BOX_ELEMENT) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int sm[]; //class_num * blockDim.x + class_num < 48k
    int *count = sm;
    int *index = sm + num_classes;
    if (threadIdx.x < num_classes) count[threadIdx.x] = 0;
    __syncthreads();
    if (idx < min((int)*bboxes, max_objects)) {
        int class_id = bboxes[1 + idx * NUM_BOX_ELEMENT + 5];
        int offset = atomicAdd(count + class_id, 1);
        index[class_id * blockDim.x + offset] = idx;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 1; i < num_classes; ++i) {
                count[i] += count[i - 1];
            }
        }
    }
    __syncthreads();
    ws += blockIdx.x * (num_classes + blockDim.x);
    if (threadIdx.x < num_classes) ws[threadIdx.x] = count[threadIdx.x];
    if (threadIdx.x < count[0])  ws[num_classes + threadIdx.x] = index[0 * blockDim.x + threadIdx.x];
    for (int class_id = 1; class_id < num_classes; ++class_id) {
        if (threadIdx.x >= count[class_id - 1] && threadIdx.x < count[class_id]) {
            ws[num_classes + threadIdx.x] = index[class_id * blockDim.x + threadIdx.x - count[class_id - 1]];
        }
    }
}

static __global__ void fast_nms_kernel_opt(
        float* __restrict__ bboxes, int * __restrict__ ws, int max_objects, int num_classes, float threshold, int NUM_BOX_ELEMENT){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int count = (int)*bboxes < max_objects ? (int)*bboxes : max_objects;
    if (idx >= count) return;
    int offset = blockIdx.x * (num_classes + blockDim.x);
    int position = ws[offset + num_classes + threadIdx.x];
    int class_id = 0;
    for (int i = 0; i < num_classes; ++i) {
        if (threadIdx.x < ws[offset + i]) {
            class_id = i;
            break;
        }
    }

    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    int valid_block = (count + blockDim.x - 1) / blockDim.x;
    for(int i = 0; i < valid_block; ++i){
        int num  = ws[i * (num_classes + blockDim.x) + class_id];
        if (class_id == 0) {
            offset = 0;
        } else {
            offset = ws[i * (num_classes + blockDim.x) + class_id - 1];
        }
        num -= offset;
        //num =  num & 127;
        for(int j = 0; j < num; ++j){
            int pitem_position = ws[i * (num_classes + blockDim.x) + num_classes + offset + j];
            float* pitem = bboxes + 1 +  pitem_position * NUM_BOX_ELEMENT;
            if(pitem_position == position) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && pitem_position < position) continue;
                float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }

            }
        }
    }

}

static __global__ void filter_kernel(
        float* __restrict__ predict, float* __restrict__ workspace,
        int num_bboxes, int num_classes, int num_tile,
        float confidence_threshold, int max_objects, int NUM_BOX_ELEMENT,
        int per_tile_bboxes){
    __shared__ int sm_cnt;
    if (threadIdx.x == 0) {
        sm_cnt = 0;
    }
    __syncthreads();

    int itile = blockIdx.x;
    int ilayer = 0;
    for (int tid=threadIdx.x; tid<per_tile_bboxes; tid+=blockDim.x) {
        int boxid = itile * per_tile_bboxes + tid;
        if (boxid >= num_bboxes)
              continue;

        int offset = boxid * (5 + num_classes);

        float* pitem = predict + offset;
        float obj_conf = pitem[4];
        if(obj_conf <= confidence_threshold)
            continue;

        float* cls_conf = pitem + 5;
        float final_conf = *cls_conf++;
        int label = 0;
        for(int i = 1; i < num_classes; ++i, ++cls_conf){
            float confidence = *cls_conf;
            if(confidence > final_conf){
                final_conf = confidence;
                label      = i;
            }
        }
        final_conf *= obj_conf;

        int index = atomicAdd(&sm_cnt, 1);
        if(index >= max_objects)
            break;

        float cx = pitem[0], cy = pitem[1], w_div2 = pitem[2] / 2.0f, h_div2 = pitem[3] / 2.0f;
        float x1 = cx - w_div2;
        float y1 = cy - h_div2;
        float x2 = cx + w_div2;
        float y2 = cy + h_div2;

        float* parray = workspace + num_tile + itile * max_objects * NUM_BOX_ELEMENT;
        float* pout_item = parray + index * NUM_BOX_ELEMENT;
        *pout_item++ = x1;
        *pout_item++ = y1;
        *pout_item++ = x2;
        *pout_item++ = y2;
        *pout_item++ = final_conf;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        workspace[itile] = sm_cnt;
    }
}

static __global__ void merge_tile_kernel(
        float* __restrict__ input, float* __restrict__ output,
        int per_tile_out_size, int max_objects, int NUM_BOX_ELEMENT){
    extern __shared__ int sm[];
    int *sm_box_cnt  = sm;;
    int *sm_out_offset = sm + NUM_TILE;
    if (threadIdx.x < NUM_TILE) {
        sm_box_cnt[threadIdx.x] = input[threadIdx.x];
        sm_out_offset[threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int box_cnt = 0;
        bool clear = false;
        for (int i=0; i<NUM_TILE; i++) {
            if (clear) {
                sm_box_cnt[i] = 0;
            }
            if (box_cnt + sm_box_cnt[i] >= max_objects) {
                sm_box_cnt[i] = max_objects - box_cnt;
                box_cnt = max_objects;
                clear = true;
            } else {
                box_cnt += sm_box_cnt[i];
            }
        }
        if (blockIdx.x == 0) {
            output[0] = box_cnt;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i=1; i<NUM_TILE; i++) {
            sm_out_offset[i] = sm_out_offset[i-1] + sm_box_cnt[i-1] * NUM_BOX_ELEMENT;
        }
    }
    __syncthreads();

    int itile = blockIdx.x;
    for (int tid=threadIdx.x; tid<sm_box_cnt[itile] * NUM_BOX_ELEMENT; tid+=blockDim.x) {
        int in_idx = NUM_TILE + itile * per_tile_out_size + tid;
        int out_idx = 1 + sm_out_offset[itile] + tid;
        output[out_idx] = input[in_idx];
    }
}


#endif /* NON_MAX_SUPPRESSION_H_ */
