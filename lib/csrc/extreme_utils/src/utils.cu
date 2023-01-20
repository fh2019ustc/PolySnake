#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "cuda_common.h"


__device__ float angle_distance(float cx, float cy, int64_t x, int64_t y, float u, float v) {
    float dx = cx - float(x);
    float dy = cy - float(y);
    float n1 = sqrt(u * u + v * v);
    float n2 = sqrt(dx * dx + dy * dy);
    float dot = u * dx + v * dy;
    float distance = dot / (n1 * n2);
    return distance;
}


__device__ float l2_distance(float cx, float cy, int64_t x, int64_t y, float u, float v) {
    float dx = float(x) + u;
    float dy = float(y) + v;
    dx = cx - dx;
    dy = cy - dy;
    return sqrt(dx * dx + dy * dy);
}

__device__ void collect_tt(
    const float* ext_hm,
    const float* vote,
    const float* ct,
    int64_t* extreme_point,
    int64_t x_min,
    int64_t x_max,
    int64_t y_min,
    int64_t radius,
    int h,
    int w
) {
    int64_t ext_x = (x_min + x_max) / 2;
    int64_t ext_y = y_min;
    float max_score = ext_hm[ext_y * w + ext_x];

    for (int64_t i = -radius; i <= radius; i++) {
        if (y_min + i < 0 || y_min + i > h - 1)
            continue;
        const float* line = &ext_hm[(y_min + i) * w];
        const float* vote_line = &vote[(y_min + i) * w * 2];
        for (int64_t j = x_min; j <= x_max; j++) {
            float score = line[j];
            float vote_x = vote_line[j * 2];
            float vote_y = vote_line[j * 2 + 1];

            if (angle_distance(ct[0], ct[1], j, y_min + i, vote_x, vote_y) < 0.9)
                continue;
            // if (l2_distance(ct[0], ct[1], x_min + i, j, vote_x, vote_y) > float(radius * 3))
            //     continue;

            if (max_score < score) {
                max_score = score;
                ext_x = j;
                ext_y = (y_min + i);
            }
        }
    }

    extreme_point[0] = ext_x;
    extreme_point[1] = ext_y;
}


__device__ void collect_ll(
    const float* ext_hm,
    const float* vote,
    const float* ct,
    int64_t* extreme_point,
    int64_t x_min,
    int64_t y_min,
    int64_t y_max,
    int64_t radius,
    int h,
    int w
) {
    int64_t ext_x = x_min;
    int64_t ext_y = (y_min + y_max) / 2;
    float max_score = ext_hm[ext_y * w + ext_x];

    for (int64_t i = -radius; i <= radius; i++) {
        if (x_min + i < 0 || x_min + i > w - 1)
            continue;
        for (int64_t j = y_min; j <= y_max; j++) {
            float score = ext_hm[j * w + x_min + i];
            float vote_x = vote[j * w * 2 + (x_min + i) * 2];
            float vote_y = vote[j * w * 2 + (x_min + i) * 2 + 1];

            if (angle_distance(ct[0], ct[1], x_min + i, j, vote_x, vote_y) < 0)
                continue;
            // if (l2_distance(ct[0], ct[1], x_min + i, j, vote_x, vote_y) > float(radius * 3))
            //     continue;

            if (max_score < score) {
                max_score = score;
                ext_x = x_min + i;
                ext_y = j;
            }
        }
    }

    extreme_point[0] = ext_x;
    extreme_point[1] = ext_y;
}


__global__ void collect_extreme_point_kernel(
    const float* ext_hm,
    const int64_t* bbox,
    const int64_t* radius,
    const float* vote,
    const float* ct,
    int64_t* extreme_point,
    int b,
    int n,
    int h,
    int w
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n * 4)
        return;

    int current_b = index / (n * 4);
    int current_n = (index - current_b * n * 4) / 4;
    int direction = index % 4;

    const float* c_ext_hm = &ext_hm[current_b * 4 * h * w + direction * h * w];
    const int64_t* c_bbox = &bbox[current_b * n * 4 + current_n * 4];
    int x_min = c_bbox[0];
    int y_min = c_bbox[1];
    int x_max = c_bbox[2];
    int y_max = c_bbox[3];
    const int64_t c_radius = radius[current_b * n + current_n];
    const float* c_vote = &vote[current_b * h * w * 2];
    const float* c_ct = &ct[current_b * n * 2 + current_n * 2];
    int64_t* c_ext_point = &extreme_point[current_b * n * 4 * 2 + current_n * 4 * 2 + direction * 2];

    if (direction == 0)
        collect_tt(c_ext_hm, c_vote, c_ct, c_ext_point, x_min, x_max, y_min, c_radius, h, w);
    else if (direction == 1)
        collect_ll(c_ext_hm, c_vote, c_ct, c_ext_point, x_min, y_min, y_max, c_radius, h, w);
    else if (direction == 2)
        collect_tt(c_ext_hm, c_vote, c_ct, c_ext_point, x_min, x_max, y_max, c_radius, h, w);
    else if (direction == 3)
        collect_ll(c_ext_hm, c_vote, c_ct, c_ext_point, x_max, y_min, y_max, c_radius, h, w);
}

at::Tensor collect_extreme_point(
    const at::Tensor& ext_hm,
    const at::Tensor& bbox,
    const at::Tensor& radius,
    const at::Tensor& vote,
    const at::Tensor& ct
) {
    AT_ASSERTM(ext_hm.type().is_cuda(), "ext_hm must be a CUDA tensor");
    AT_ASSERTM(bbox.type().is_cuda(), "bbox must be a CUDA tensor");
    AT_ASSERTM(radius.type().is_cuda(), "radius must be a CUDA tensor");
    AT_ASSERTM(vote.type().is_cuda(), "vote must be a CUDA tensor");
    AT_ASSERTM(ct.type().is_cuda(), "ct must be a CUDA tensor");

    // b, 4, h, w
    auto b = ext_hm.size(0);
    auto h = ext_hm.size(2);
    auto w = ext_hm.size(3);

    // b, n, 4
    AT_ASSERTM(b == bbox.size(0), "bbox must have the same batch size with ext_hm");
    auto n = bbox.size(1);

    // b, h, w, 2
    AT_ASSERTM(b == vote.size(0), "vote must have the same batch size with ext_hm");
    AT_ASSERTM(h == vote.size(1), "vote must have the same batch size with ext_hm");
    AT_ASSERTM(w == vote.size(2), "vote must have the same batch size with ext_hm");

    // b, n, 2
    AT_ASSERTM(b == ct.size(0), "ct must have the same batch size with bbox");
    AT_ASSERTM(n == ct.size(1), "ct must have the same batch size with bbox");

    // b, n, 4, 2
    auto extreme_point = at::zeros({b, n, 4, 2}, bbox.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n * 4, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    collect_extreme_point_kernel<<<bdim, tdim, 0, stream>>>(
        ext_hm.contiguous().data<float>(),
        bbox.contiguous().data<int64_t>(),
        radius.contiguous().data<int64_t>(),
        vote.contiguous().data<float>(),
        ct.contiguous().data<float>(),
        extreme_point.data<int64_t>(),
        b,
        n,
        h,
        w
    );
    THCudaCheck(cudaGetLastError());

    return extreme_point;
}


__global__ void _calculate_edge_num(
    int64_t* edge_num,
    const int64_t* edge_num_sum,
    const int64_t* edge_idx_sort,
    const int b,
    const int n,
    const int orig_p_num,
    const int64_t p_num
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n)
        return;

    const int c_b = index / n;
    const int c_n = index % n;

    int64_t* c_edge_num = &edge_num[c_b * n * orig_p_num + c_n * orig_p_num];
    const int64_t c_edge_num_sum = edge_num_sum[c_b * n + c_n];
    const int64_t* c_edge_idx_sort = &edge_idx_sort[c_b * n * orig_p_num + c_n * orig_p_num];

    if (c_edge_num_sum == p_num)
        return;

    if (c_edge_num_sum < p_num)
        c_edge_num[c_edge_idx_sort[0]] += p_num - c_edge_num_sum;
    else {
        int id = 0;
        int64_t pass_num = c_edge_num_sum - p_num;
        while (pass_num > 0) {
            int64_t edge_idx = c_edge_idx_sort[id];
            if (c_edge_num[edge_idx] > pass_num) {
                c_edge_num[edge_idx] -= pass_num;
                pass_num = 0;
            } else {
                pass_num -= c_edge_num[edge_idx] - 1;
                c_edge_num[edge_idx] = 1;
                id += 1;
            }
        }
    }
}

void calculate_edge_num(
    at::Tensor& edge_num,
    const at::Tensor& edge_num_sum,
    const at::Tensor& edge_idx_sort,
    const int p_num
) {
    AT_ASSERTM(edge_num.type().is_cuda(), "edge_num must be a CUDA tensor");
    AT_ASSERTM(edge_num_sum.type().is_cuda(), "edge_num_sum must be a CUDA tensor");
    AT_ASSERTM(edge_idx_sort.type().is_cuda(), "edge_idx_sort must be a CUDA tensor");

    // b, n, orig_p_num
    auto b = edge_num.size(0);
    auto n = edge_num.size(1);
    auto orig_p_num = edge_num.size(2);
    AT_ASSERTM(edge_num.is_contiguous(), "edge_num must be contiguous");

    // b, n
    AT_ASSERTM(b == edge_num_sum.size(0), "edge_num_sum must have the same batch size with edge_num");
    AT_ASSERTM(n == edge_num_sum.size(1), "edge_num_sum must have the same batch size with edge_num");

    // b, n, orig_p_num
    AT_ASSERTM(b == edge_idx_sort.size(0), "edge_idx_sort must have the same batch size with edge_num");
    AT_ASSERTM(n == edge_idx_sort.size(1), "edge_idx_sort must have the same batch size with edge_num");
    AT_ASSERTM(orig_p_num == edge_idx_sort.size(2), "edge_idx_sort must have the same batch size with edge_num");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    _calculate_edge_num<<<bdim, tdim, 0, stream>>>(
        edge_num.data<int64_t>(),
        edge_num_sum.contiguous().data<int64_t>(),
        edge_idx_sort.contiguous().data<int64_t>(),
        b,
        n,
        orig_p_num,
        int64_t(p_num)
    );
    THCudaCheck(cudaGetLastError());
}


__global__ void _calculate_wnp(
    const int64_t* edge_num,
    const int64_t* edge_start_idx,
    float* weight,
    int64_t* ind,
    const int b,
    const int n,
    const int orig_p_num,
    const int p_num
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n * orig_p_num)
        return;

    const int c_b = index / (n * orig_p_num);
    const int c_n = (index - c_b * n * orig_p_num) / orig_p_num;
    const int c_edge_idx = index % orig_p_num;

    const int64_t c_edge_num = edge_num[index];
    const int c_start_idx = int(edge_start_idx[index]);
    float* c_weight = &weight[c_b * n * p_num + c_n * p_num + c_start_idx];
    int64_t* c_ind = &ind[c_b * n * p_num * 2 + c_n * p_num * 2 + c_start_idx * 2];

    for (int64_t i = 0; i < c_edge_num; i++) {
        c_weight[i] = float(i) / float(c_edge_num);
        c_ind[i * 2] = int64_t(c_edge_idx);
        c_ind[i * 2 + 1] = int64_t((c_edge_idx + 1) % orig_p_num);
    }
}

std::tuple<at::Tensor, at::Tensor> calculate_wnp(
    const at::Tensor& edge_num,
    const at::Tensor& edge_start_idx,
    const int p_num
) {
    AT_ASSERTM(edge_num.type().is_cuda(), "edge_num must be a CUDA tensor");

    // b, n, orig_p_num
    auto b = edge_num.size(0);
    auto n = edge_num.size(1);
    auto orig_p_num = edge_num.size(2);

    // b, n, orig_p_num
    AT_ASSERTM(b == edge_start_idx.size(0), "edge_start_idx must have the same batch size with edge_num");
    AT_ASSERTM(n == edge_start_idx.size(1), "edge_start_idx must have the same batch size with edge_num");
    AT_ASSERTM(orig_p_num == edge_start_idx.size(2), "edge_start_idx must have the same batch size with edge_num");

    auto weight = at::zeros({b, n, p_num, 1}, edge_num.options().dtype(at::kFloat));
    auto ind = at::zeros({b, n, p_num, 2}, edge_num.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n * orig_p_num, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    _calculate_wnp<<<bdim, tdim, 0, stream>>>(
        edge_num.contiguous().data<int64_t>(),
        edge_start_idx.contiguous().data<int64_t>(),
        weight.data<float>(),
        ind.data<int64_t>(),
        b,
        n,
        orig_p_num,
        p_num
    );
    THCudaCheck(cudaGetLastError());

    return std::make_tuple(weight, ind);
}


__global__ void _roll_array(
    const float* array,
    const int64_t* step,
    float* new_array,
    const int b,
    const int n,
    const int d
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= b * n * d)
        return;

    const int c_b = index / (n * d);
    const int c_n = (index - c_b * n * d) / d;
    const int c_d = index % d;

    const float c_array_element = array[c_b * n * d + c_n * d + c_d];
    float* c_new_array = &new_array[c_b * n * d];

    int c_step = int(step[c_b]);
    int new_n = ((c_n + c_step) % n + n) % n;
    int position = new_n * d + c_d;

    c_new_array[position] = c_array_element;
}


at::Tensor roll_array(
    const at::Tensor& array,
    const at::Tensor& step
) {
    AT_ASSERTM(array.type().is_cuda(), "array must be a CUDA tensor");
    AT_ASSERTM(step.type().is_cuda(), "step must be a CUDA tensor");

    // b, n, d
    auto b = array.size(0);
    auto n = array.size(1);
    auto d = array.size(2);

    // b
    AT_ASSERTM(b == step.size(0), "step must have the same batch size with array");

    auto new_array = at::zeros_like(array);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdim0, bdim1, bdim2;
    int tdim0, tdim1, tdim2;
    getGPULayout(b * n * d, 1, 1, &bdim0, &bdim1, &bdim2, &tdim0, &tdim1, &tdim2);
    dim3 bdim(bdim0, bdim1, bdim2);
    dim3 tdim(tdim0, tdim1, tdim2);

    _roll_array<<<bdim, tdim, 0, stream>>>(
        array.contiguous().data<float>(),
        step.contiguous().data<int64_t>(),
        new_array.data<float>(),
        b,
        n,
        d
    );
    THCudaCheck(cudaGetLastError());

    return new_array;
}
