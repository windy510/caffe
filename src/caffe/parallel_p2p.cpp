#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <sstream>
#include <glog/logging.h>

#include <caffe/caffe.hpp>
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"

#ifndef CPU_ONLY
#include <cuda_runtime.h>

using namespace std;

namespace caffe {

template<typename Dtype>
P2PSync<Dtype>::P2PSync(const GPUParams<Dtype>& params)
    : params_(params),
      chunks_(chunks(params.params().len_used())),
      calls_("calls", CHUNK * sizeof(Dtype)),
      cycles_("cycles") {

  size_t size = params.params().len_buff() * sizeof(Dtype);
  Dtype* gpu = params.gpu();
  CUDA_CHECK(cudaMalloc((void** ) &gpu_last_, size));
  CUDA_CHECK(cudaMemcpy(gpu_last_, gpu, size, cudaMemcpyDeviceToDevice));
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
  stop();
  CUDA_CHECK(cudaFree((void* ) gpu_last_));
}

template<typename Dtype>
void P2PSync<Dtype>::run() {
  CUDA_CHECK(cudaSetDevice(this->params_.device()));
  GPUStream<Dtype> gpu_stream;
  const cudaStream_t& stream = gpu_stream.stream();

  // Current cpu values when invoking kernel, gradients on the way back
  Dtype* buf;
  Dtype* tmp;
  CUDA_CHECK(cudaMalloc((void** ) &buf, CHUNK * sizeof(Dtype)));
  CaffeMallocHost((void**) &tmp, CHUNK * sizeof(Dtype));

  const size_t len = CHUNK * sizeof(Dtype);
  // Explicit directions for readability
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  const cudaMemcpyKind get = cudaMemcpyDeviceToHost;
  uint32_t index = 0;
  Dtype* cpu = this->params_.params().cpu();
  Dtype* gpu = this->params_.gpu();
  Dtype* last = this->gpu_last_;
  uint8_t get_grads = true;

  while (!must_stop()) {
    size_t off = index * CHUNK;
    CUDA_CHECK(cudaMemcpyAsync(buf, &cpu[off], len, put, stream));
    // TODO simpler kernel
    sync_worker_kernel<Dtype>(gpu, last, &buf, &off, &buf, &get_grads,  //
                              0, 1, stream, CHUNK);
    CUDA_CHECK(cudaMemcpyAsync(tmp, buf, len, get, stream));
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < CHUNK; ++i)
      cpu[off + i] += tmp[i];
    if (++index == chunks_) {
      index = 0;
      cycles_++;
    }
    calls_++;
  }

  CaffeFreeHost((void*) tmp);
  CUDA_CHECK(cudaFree((void* ) buf));
}

INSTANTIATE_CLASS(P2PSync);
}

#endif
