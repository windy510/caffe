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
    : barrier_(new boost::barrier(params.size())),
      gpus_(params.size()),
      sent_("sent", CHUNK * sizeof(Dtype)),
      cycles_("cycles") {
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
    gpus_[i].reset(new GPU(*this, params, i));
    sent_.add_child(&(gpus_[i]->sent_));
    cycles_.add_child(&(gpus_[i]->cycles_));
  }
}

template<typename Dtype>
void P2PSync<Dtype>::start() {
  for (int i = 0; i < gpus_.size(); ++i) {
    gpus_[i].get()->start();
  }
}

template<typename Dtype>
void P2PSync<Dtype>::stop() {
  for (int i = 0; i < gpus_.size(); ++i) {
    gpus_[i].get()->stop(false);
  }
  for (int i = 0; i < gpus_.size(); ++i) {
    gpus_[i].get()->stop(true);
  }
}

//

template<typename Dtype>
P2PSync<Dtype>::Multicast::Multicast(const GPU& gpu)
    : index_(gpu.index()),
      source_(),
      targets_(gpu.params().size()),
      callbacks_(gpu.params().size()),
      pending_targets_(1),
      chunk_() {
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  CUDA_CHECK(cudaSetDevice(gpu.params()[index_]->device()));
  CUDA_CHECK(cudaMalloc((void** ) &source_, CHUNK * sizeof(Dtype)));

  for (int i = 0; i < targets_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaSetDevice(gpu.params()[i]->device()));
      CUDA_CHECK(cudaMalloc((void** ) &targets_[i], CHUNK * sizeof(Dtype)));
    }
    callbacks_[i].queue_ = &gpu.sync().gpus()[i]->queue_;
    callbacks_[i].multicast_ = this;
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
P2PSync<Dtype>::Multicast::~Multicast() {
  for (int i = 0; i < targets_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaFree((void* ) targets_[i]));
    }
  }
  CUDA_CHECK(cudaFree((void* ) source_));
}

//

template<typename Dtype>
P2PSync<Dtype>::GPU::GPU(const P2PSync& sync,
                         const vector<GPUParams<Dtype>*> params, int index)
    : sync_(sync),
      params_(params),
      index_(index),
      chunks_(chunks(params[0]->len_used())),
      sent_("gpu " + lexical_cast<string>(params[index]->device()),
            CHUNK * sizeof(Dtype)),
      sent_to_each_(params.size()),
      cycles_("cycles") {

  // Perf counters
  for (int i = 0; i < params.size(); ++i) {
    string peer(lexical_cast<string>(params[i]->device()));
    Meter* m = new Meter("-> " + peer, CHUNK * sizeof(Dtype));
    sent_.add_child(m);
    sent_to_each_[i].reset(m);
  }
}

template<typename Dtype>
P2PSync<Dtype>::GPU::~GPU() {
  // All sync threads must have been stopped earlier
  CHECK(must_stop());
}

template<typename Dtype>
void P2PSync<Dtype>::GPU::GPU::run() {
  const int device = params_[index_]->device();
  CUDA_CHECK(cudaSetDevice(device));

  // Enable p2p access to each device
  for (int i = 0; i < params_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(params_[i]->device(), 0));
    }
  }

  // Create async stream
  cudaStream_t stream;
  //  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int least, greatest;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least);

  // Allocate weight copy to measure gradient
  size_t size = params_[index_]->len_buff() * sizeof(Dtype);
  Dtype* copy;
  CUDA_CHECK(cudaMalloc((void** ) &copy, size));

  // Copy current weights
  Dtype* data = params_[index_]->data();
  // Explicit for debug purposes
  cudaMemcpyKind dev2dev = cudaMemcpyDeviceToDevice;
  CUDA_CHECK(cudaMemcpy(copy, data, size, dev2dev));

  // Create queues and buffers
  for (int i = 0; i < QUEUE; ++i) {
    queue_.push(new Multicast(*this));
  }

  sync_.barrier_->wait();
  uint32_t chunk = 0;

  try {
    while (!must_stop()) {
      Multicast* m = queue_.pop();

      if (m->index_ == index_) {  // Multicast is on initial GPU
        // Wait until all targets are done with their buffer
        if (--m->pending_targets_ == 0) {
          // Measure gradient
          size_t offset = chunk * CHUNK;
          p2p_sync_send<Dtype>(data, copy, offset, m->source_, stream);
          m->chunk_ = chunk;

          // Send gradient
          m->pending_targets_ = sync_.gpus_.size() - 1;
          for (int i = 0; i < sync_.gpus_.size(); ++i) {
            if (i != index_) {
              CUDA_CHECK(
                  cudaMemcpyAsync(m->targets_[i], m->source_,  //
                                  CHUNK * sizeof(Dtype), dev2dev, stream));
              Callback* c = &m->callbacks_[i];
              cudaStreamAddCallback(stream, add_to_queue, (void*) c, 0);
              sent_to_each_[i]->tick();
            }
          }
          if (++chunk == chunks_) {
            chunk = 0;
            cycles_.tick();
          }
        }
      } else {  // Multicast arrived at a target
        // Apply gradient
        size_t offset = m->chunk_ * CHUNK;
        p2p_sync_recv<Dtype>(data, copy, offset, m->targets_[index_], stream);
        Callback* c = &m->callbacks_[m->index_];
        cudaStreamAddCallback(stream, add_to_queue, (void*) c, 0);
      }
    }
    throw boost::thread_interrupted();
  } catch (...) {
    // Wait for callbacks
    cudaStreamSynchronize(stream);
    sync_.barrier_->wait();
    Multicast* m;
    // Put back all m on their own GPU
    vector<Multicast*> tmp;
    while (queue_.try_pop(m)) {
      tmp.push_back(m);
    }
    sync_.barrier_->wait();
    for (int i = 0; i < tmp.size(); ++i) {
      m = tmp[i];
      sync_.gpus_[m->index_]->queue_.push(m);
    }
    sync_.barrier_->wait();
    // Free only once
    std::set<Multicast*> set;
    while (queue_.try_pop(m)) {
      set.insert(m);
    }
    for (typename std::set<Multicast*>::iterator it = set.begin();
        it != set.end(); ++it) {
      delete *it;
    }
    sync_.barrier_->wait();

    CUDA_CHECK(cudaFree((void* ) copy));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < params_.size(); ++i) {
      if (i != index_) {
        CUDA_CHECK(cudaDeviceDisablePeerAccess(params_[i]->device()));
      }
    }
  }
}

template<typename Dtype>
void CUDART_CB P2PSync<Dtype>::GPU::GPU::add_to_queue(cudaStream_t stream,
                                                      cudaError_t status,
                                                      void* data) {
  Callback* c = (Callback*) data;
  c->queue_->push(c->multicast_);
}

INSTANTIATE_CLASS(P2PSync);
}

#endif
