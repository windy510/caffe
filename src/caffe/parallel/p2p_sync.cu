#include <cuda_runtime.h>
#include <stdio.h>
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Dtype>
__global__
void p2p_sync_send(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  Dtype d = gpu[off + i] - last[off + i];
  gpu[off + i] = last[off + i] = chunk[i] + d;
  chunk[i] = d;
}

template<typename Dtype>
__global__
void p2p_sync_recv(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  Dtype d = gpu[off + i] - last[off + i];
  gpu[off + i] = last[off + i] = chunk[i] + d;
  chunk[i] = d;
}

template<typename Dtype>
void p2p_sync_send(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off, cudaStream_t& stream) {
  int threadsPerBlock = 256; // TODO bench
  int blocksPerGrid = GPUSync<Dtype>::CHUNK / threadsPerBlock;
  GPUSyncKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(gpu, last, chunk, off);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void p2p_sync_recv(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off, cudaStream_t& stream) {
  int threadsPerBlock = 256; // TODO bench
  int blocksPerGrid = GPUSync<Dtype>::CHUNK / threadsPerBlock;
  GPUSyncKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(gpu, last, chunk, off);
  CUDA_POST_KERNEL_CHECK;
}

template void p2p_sync_send<float>(float* gpu, float* last, float* chunk, size_t off, cudaStream_t& stream);
template void p2p_sync_send<double>(double* gpu, double* last, double* chunk, size_t off, cudaStream_t& stream);
template void p2p_sync_recv<float>(float* gpu, float* last, float* chunk, size_t off, cudaStream_t& stream);
template void p2p_sync_recv<double>(double* gpu, double* last, double* chunk, size_t off, cudaStream_t& stream);
}
