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

// From CUDA samples

inline bool IsGPUCapableP2P(cudaDeviceProp* pProp) {
#ifdef _WIN32
  return (bool)(pProp->tccDriver ? true : false);
#else
  return (bool) (pProp->major >= 2);
#endif
}

inline bool IsAppBuiltAs64() {
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
  return 1;
#else
  return 0;
#endif
}

using namespace std;

namespace caffe {

template<typename Dtype>
P2PSync<Dtype>::P2PSync(const vector<GPUParams<Dtype>*> params)
    : calls_("calls", CHUNK * sizeof(Dtype)),
      cycles_("cycles") {

  for (int i = 1; i < params.size(); ++i) {
    CHECK(params[i].params().len_used() == params[0].params().len_used());
    CHECK(params[i].params().len_buff() == params[0].params().len_buff());
  }
  CHECK(IsAppBuiltAs64()) << ("P2PSync is only supported with on 64-bit OS");
  for (int i = 0; i < params.size(); ++i) {
    const int dev = params[i].device();
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(IsGPUCapableP2P(&prop)) << "GPU " << dev << " does not support P2P";
    CHECK(prop.unifiedAddressing) << "GPU " << dev << " does not support UVA";
  }

  emits_.resize(params.size());
  for (int i = 0; i < params.size(); ++i) {
    emits_[i].get().reset(new P2PEmit(params, i));
  }
}

template<typename Dtype>
P2PSync<Dtype>::P2PEmit::P2PEmit(const P2PSync& sync,
                                 const vector<GPUParams<Dtype>*> params,
                                 int index)
    : sync_(sync),
      params_(params),
      index_(index),
      chunks_(chunks(params[0].params().len_used())) {

  for (int i = 0; i < params.size(); ++i) {
    if (i != index) {
      const int device = params[index].device();
      const int peer = params[i].device();
      int access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
      CHECK(access) << "GPU " << device << " cannot access GPU " << peer;
    }
  }
}

template<typename Dtype>
P2PSync<Dtype>::P2PEmit::~P2PEmit() {
  stop();
}

template<typename Dtype>
void P2PSync<Dtype>::P2PEmit::P2PEmit::run() {
  // Create a stream for each device
  vector<cudaStream_t> streams(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(this->params_[i].device()));
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  // Enable p2p access to each device
  CUDA_CHECK(cudaSetDevice(this->params_[index_].device()));
  for (int i = 0; i < params_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(this->params_[i].device(), 0));
    }
  }

  // Allocate weight copy to measure gradient
  size_t size = params_[index_].params().len_buff() * sizeof(Dtype);
  Dtype* buffer = params_[index_].buffer();
  Dtype* copy;
  CUDA_CHECK(cudaMalloc((void** ) &copy, size));
  CUDA_CHECK(cudaMemcpy(copy, buffer, size, cudaMemcpyDeviceToDevice));

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
    cudaStreamSynchronize (stream);
    for (size_t i = 0; i < CHUNK; ++i)
      cpu[off + i] += tmp[i];
    if (++index == chunks_) {
      index = 0;
      cycles_++;
    }
    calls_++;
  }

  CUDA_CHECK(cudaFree((void* ) copy));
  for (int i = 0; i < params_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(this->params_[i].device()));
    }
  }
  for (int i = 0; i < params_.size(); ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

INSTANTIATE_CLASS(P2PSync);
}

#endif
