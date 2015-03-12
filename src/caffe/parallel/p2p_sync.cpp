#include <boost/thread.hpp>
#include <cstdlib>
#include <cstring>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"

#ifndef CPU_ONLY
#include <cuda_runtime.h>

using boost::lexical_cast;
using namespace std;

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

namespace caffe {

template<typename Dtype>
P2PSync<Dtype>::P2PSync(const vector<GPUParams<Dtype>*>& params)
    : params_(params),
      gpus_(params.size()) {
  for (int i = 1; i < params.size(); ++i) {
    CHECK(params[i]->len_used() == params[0]->len_used());
    CHECK(params[i]->len_buff() == params[0]->len_buff());
  }
  CHECK(IsAppBuiltAs64()) << ("P2PSync is only supported on a 64-bit OS");
  for (int i = 0; i < params.size(); ++i) {
    const int dev = params[i]->device();
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(IsGPUCapableP2P(&prop)) << "GPU " << dev << " does not support P2P";
    CHECK(prop.unifiedAddressing) << "GPU " << dev << " does not support UVA";
  }
  // TODO supposed to work, cuda going through the host when no p2p, but crashes
  for (int i = 0; i < params.size(); ++i) {
    for (int j = 0; j < params.size(); ++j) {
      if (j != i) {
        const int device = params[i]->device();
        const int peer = params[j]->device();
        int access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
        CHECK(access) << "GPU " << device << " cannot access GPU " << peer;
      }
    }
  }
  for (int i = 0; i < params.size(); ++i) {
    gpus_[i].reset(new GPU(*this, i));
  }
}

template<typename Dtype>
void P2PSync<Dtype>::send(int index) const {
  gpus_[index].get()->send();
}

template<typename Dtype>
void P2PSync<Dtype>::sum_targets() {
  GPU* gpu = gpus_[0].get();
  for (int i = 1; i < gpu->targets_.size(); ++i) {
    caffe_gpu_add(params_[0]->len_used(),  //
        gpu->targets_[i],  //
        params_[0]->diff(),  //
        params_[0]->diff());
  }
}

//

template<typename Dtype>
P2PSync<Dtype>::GPU::GPU(const P2PSync& sync, int index)
    : sync_(sync),
      params_(*(sync.params()[index])),
      index_(index),
      targets_(sync.params().size()) {

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  for (int i = 0; i < sync.params().size(); ++i) {
    if (i != index) {
      const int device = params_.device();
      const int peer = sync.params()[i]->device();

      // Enable p2p access to each device
      int access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
      if (access) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
      } else {
        LOG(INFO)<< "GPUs " << device << " and " << peer << " go through host.";
      }

      // Allocate receiving buffer
      CUDA_CHECK(cudaSetDevice(peer));
      CUDA_CHECK(cudaMalloc((void** ) &targets_[i],  //
          params_.len_buff() * sizeof(Dtype)));
    }
  }

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
P2PSync<Dtype>::GPU::~GPU() {
  CUDA_CHECK(cudaStreamDestroy(stream_));
  for (int i = 0; i < sync_.params().size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaFree(targets_[i]));

      const int device = params_.device();
      const int peer = sync_.params()[i]->device();
      int access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
      if (access) {
        CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::GPU::GPU::send() {
  // Needed if non blocking stream
  cudaStreamSynchronize(cudaStreamDefault);

  if (index_ == 0) {
    for (int i = 1; i < sync_.gpus_.size(); ++i) {
      CUDA_CHECK(cudaMemcpyAsync(  //
          sync_.params_[i]->data(),  //
          sync_.params_[0]->data(),  //
          sync_.params_[0]->len_used() * sizeof(Dtype),  //
          cudaMemcpyDeviceToDevice, stream_));
    }
  } else {
    CUDA_CHECK(cudaMemcpyAsync(  //
        sync_.gpus_[0]->targets_[index_],  //
        sync_.params_[index_]->diff(),  //
        sync_.params_[index_]->len_used() * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, stream_));
  }
  cudaStreamSynchronize(stream_);
}

INSTANTIATE_CLASS(P2PSync);
}  // namespace caffe

#endif
