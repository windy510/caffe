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
P2PSync<Dtype>::P2PSync(const vector<GPUParams<Dtype>*>& params) {
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

  gpus_.resize(params.size());
  for (int i = 0; i < params.size(); ++i) {
    gpus_[i].reset(new GPU(*this, params, i));
  }

  // Add each send_channel to corresponding recv channels list
  for (int source = 0; source < gpus_.size(); ++source) {
    vector<Channel*>& send = gpus_[source].get()->send_channels_;
    for (int i = 0; i < send.size(); ++i) {
      Channel* channel = send[i];
      for (int target = 0; target < gpus_.size(); ++target) {
        CHECK(channel->source_device_ == params[source]->device());
        if (channel->target_device_ == params[target]->device()) {
          gpus_[target].get()->recv_channels_.push_back(channel);
          gpus_[target].get()->recv_.add_child(&channel->recv_);
        }
      }
    }
  }
  for (int i = 0; i < gpus_.size(); ++i) {
    CHECK(gpus_[i].get()->recv_channels_.size() == gpus_.size() - 1);
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
    gpus_[i].get()->stop();
  }
}

//

template<typename Dtype>
P2PSync<Dtype>::Multicast::Multicast(int source_device,
                                     vector<int> target_devices)
    : chunk_() {
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const unsigned int flags = cudaEventDisableTiming;

  CUDA_CHECK(cudaSetDevice(source_device));
  CUDA_CHECK(cudaMalloc((void** ) &source_, CHUNK * sizeof(Dtype)));
  CUDA_CHECK(cudaEventCreateWithFlags(&source_event_, flags));

  for (int i = 0; i < target_devices.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(target_devices[i]));
    CUDA_CHECK(cudaMalloc((void** ) &targets_[i], CHUNK * sizeof(Dtype)));
    CUDA_CHECK(cudaEventCreateWithFlags(&target_events_[i], flags));
  }
  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
P2PSync<Dtype>::Multicast::~Multicast() {
  for (int i = 0; i < targets_.size(); ++i) {
    CUDA_CHECK(cudaEventDestroy(target_events_[i]));
    CUDA_CHECK(cudaFree((void** ) &targets_[i]));
  }
  CUDA_CHECK(cudaEventDestroy(source_event_));
  CUDA_CHECK(cudaFree((void** ) &source_));
}

template<typename Dtype>
bool P2PSync<Dtype>::Multicast::target_events_done() const {
  for (int i = 0; i < target_events_.size(); ++i) {
    if (cudaEventQuery(target_events_[i]) != cudaSuccess) {
      return false;
    }
  }
  return true;
}

target_events_done

//

template<typename Dtype>
P2PSync<Dtype>::Channel::Channel(int source_device, int target_device)
    : source_device_(source_device),
      target_device_(target_device),
      sent_(
          "sent " + lexical_cast < string
              > (source_device) + "->" + lexical_cast < string
              > (target_device),
          CHUNK * sizeof(Dtype)),
      recv_(
          "recv " + lexical_cast < string
              > (source_device) + "<-" + lexical_cast < string
              > (target_device),
          CHUNK * sizeof(Dtype)) {
  for (int i = 0; i < LENGTH; ++i) {
    free_.push(new Message(source_device, target_device));
  }
}

template<typename Dtype>
P2PSync<Dtype>::Channel::~Channel() {
  Message* message;
  while (free_.try_pop(message)) {
    delete message;
  }
  while (full_.try_pop(message)) {
    delete message;
  }
}

//

template<typename Dtype>
P2PSync<Dtype>::GPU::GPU(const P2PSync& sync,
                         const vector<GPUParams<Dtype>*> params, int index)
    : sync_(sync),
      params_(params),
      index_(index),
      chunks_(chunks(params[0]->len_used())),
      sent_("sent", CHUNK * sizeof(Dtype)),
      recv_("recv", CHUNK * sizeof(Dtype)),
      cycles_("cycles") {

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int device = params[index]->device();
  CUDA_CHECK(cudaSetDevice(device));

  for (int i = 0; i < params.size(); ++i) {
    if (i != index) {
      const int peer = params[i]->device();

      // Enable p2p access to each device
      int access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&access, device, peer));
      CHECK(access) << "GPU " << device << " cannot access GPU " << peer;
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));

      // Create channels
      Channel* channel = new Channel(device, peer);
      send_channels_.push_back(channel);
      sent_.add_child(&channel->sent_);
    }
  }
  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
P2PSync<Dtype>::GPU::~GPU() {
  // All sync threads must have been stopped earlier
  CHECK(must_stop());

  for (int i = 0; i < send_channels_.size(); ++i) {
    delete send_channels_[i];
  }
}

template<typename Dtype>
void P2PSync<Dtype>::GPU::GPU::run() {
  CUDA_CHECK(cudaSetDevice(params_[index_]->device()));
  const int device = params_[index_]->device();
  if (device) {
  }  // TODO just for debug

  // Create async stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Allocate weight copy to measure gradient
  size_t size = params_[index_]->len_buff() * sizeof(Dtype);
  Dtype* copy;
  CUDA_CHECK(cudaMalloc((void** ) &copy, size));

  // Explicit for debug purposes
  Dtype* data = params_[index_]->data();
  cudaMemcpyKind dev2dev = cudaMemcpyDeviceToDevice;
  CUDA_CHECK(cudaMemcpy(copy, data, size, dev2dev));

  deque<Multicast*> free, sending, sent;
  uint32_t chunk = 0;

  // Dtype* hist = this->params_.hist();
  // sleep(10000);

  while (!must_stop()) {
    Multicast* m;

    if (!free.empty()) {
      m = free.front();
      if (m->target_events_done()) {
        if (params_[index_]->device() >= 0) {
          free.pop();
          size_t offset = chunk * CHUNK;
          p2p_sync_send<Dtype>(data, copy, offset, m->source_, stream);
          m->chunk_ = chunk;
          CUDA_CHECK(cudaEventRecord(m->source_event_, stream));
          sending.push_back(m);
          if (++chunk == chunks_) {
            chunk = 0;
            cycles_++;
          }
        }
      }
    }

    if (!sending.empty()) {
      m = free.front();
      if(cudaEventQuery(message->target_done_) == cudaSuccess) {

      if (m->target_events_done()) {
      }
    }

    if (all_ready_to_send) {
      for (int i = 0; i < send_channels_.size(); ++i) {
        Channel& channel = *(send_channels_[i]);
        Message* message;

        // Compute data to send when a free buffer is available
        if (channel.free_.try_peek(message)
            && cudaEventQuery(message->target_done_) == cudaSuccess) {
        }
        // Send message to target
        if (!channel.pending_.empty()) {
          message = channel.pending_.front();
          if (cudaEventQuery(message->source_done_) == cudaSuccess) {
            channel.pending_.pop_front();
            CUDA_CHECK(cudaMemcpyAsync(message->target_, message->source_,  //
                                       CHUNK * sizeof(Dtype), dev2dev, stream));
            CUDA_CHECK(cudaEventRecord(message->source_done_, stream));
            channel.full_.push(message);
            channel.sent_++;
          }
        }
      }
    }

    for (int i = 0; i < recv_channels_.size(); ++i) {
      Channel& channel = *(recv_channels_[i]);
      Message* message;

      // Receive message when async transfer is done
      if (channel.full_.try_peek(message)) {
        int q = cudaEventQuery(message->source_done_);
        CHECK(q == cudaSuccess || q == cudaErrorNotReady);
        if (q == cudaSuccess) {
          channel.full_.pop();

          size_t offset = message->chunk_ * CHUNK;
          p2p_sync_recv<Dtype>(data, copy, offset, message->target_, stream);
          CUDA_CHECK(cudaEventRecord(message->target_done_, stream));
          channel.free_.push(message);
          channel.recv_++;
        }
      }
    }
  }

  CUDA_CHECK(cudaFree((void* ) copy));
  for (int i = 0; i < params_.size(); ++i) {
    if (i != index_) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(params_[i]->device()));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

INSTANTIATE_CLASS(P2PSync);
}

#endif
