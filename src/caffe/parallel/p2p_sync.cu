#include <cuda_runtime.h>
#include <stdio.h>
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Dtype>
__global__
void p2p_sync_send(Dtype* data, Dtype* copy, size_t off, Dtype* chunk) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  chunk[i] = data[off + i] - copy[off + i];
  copy[off + i] = data[off + i];
//  if (off == 0 && i == 1000) {
//    printf("send data  %f\n", data[off + i]);
//    printf("send copy  %f\n", copy[off + i]);
//    printf("send chunk %f\n", chunk[i]);
//  }
}

template<typename Dtype>
__global__
void p2p_sync_recv(Dtype* data, Dtype* copy, size_t off, Dtype* chunk) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  data[off + i] += chunk[i];
  copy[off + i] += chunk[i];
//  if (off == 0 && i == 1000) {
//    printf("recv data  %f\n", data[off + i]);
//    printf("recv copy  %f\n", copy[off + i]);
//    printf("recv chunk %f\n", chunk[i]);
//  }
}

template<typename Dtype>
void p2p_sync_send(Dtype* data, Dtype* copy, size_t off, Dtype* chunk,
                   const cudaStream_t& stream) {
  int threadsPerBlock = 256;  // TODO bench
  int blocksPerGrid = P2PSync<Dtype>::CHUNK / threadsPerBlock;
  p2p_sync_send<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(  //
      data, copy, off, chunk);
  CUDA_POST_KERNEL_CHECK
  ;
}

template<typename Dtype>
void p2p_sync_recv(Dtype* data, Dtype* copy, size_t off, Dtype* chunk,
                   const cudaStream_t& stream) {
  int threadsPerBlock = 256;  // TODO bench
  int blocksPerGrid = P2PSync<Dtype>::CHUNK / threadsPerBlock;
  p2p_sync_recv<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(  //
      data, copy, off, chunk);
  CUDA_POST_KERNEL_CHECK
  ;
}

template void p2p_sync_send<float>(float* data, float* copy, size_t off,
                                   float* chunk, const cudaStream_t& stream);
template void p2p_sync_send<double>(double* data, double* copy, size_t off,
                                    double* chunk, const cudaStream_t& stream);
template void p2p_sync_recv<float>(float* data, float* copy, size_t off,
                                   float* chunk, const cudaStream_t& stream);
template void p2p_sync_recv<double>(double* data, double* copy, size_t off,
                                    double* chunk, const cudaStream_t& stream);
}
