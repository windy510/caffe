#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_[i].data_.mutable_cpu_data();
  if (this->output_labels_)
    for(int i = 0; i < PREFETCH_COUNT; ++i)
      prefetch_[i].label_.mutable_cpu_data();

  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      CUDA_CHECK(cudaGetDevice(&device_));
      for(int i = 0; i < PREFETCH_COUNT; ++i)
        prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_)
        for(int i = 0; i < PREFETCH_COUNT; ++i)
          prefetch_[i].label_.mutable_gpu_data();
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if(device_ >= 0) {
    CUDA_CHECK(cudaSetDevice(device_));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
#endif

  while(!must_stop()) {
    Batch<Dtype>* batch = free_.pop();
    load_batch(batch);
#ifndef CPU_ONLY
    if(device_ >= 0) {
      batch->data_.data().get()->async_gpu_push(stream);
      cudaStreamSynchronize(stream);
    }
#endif
    full_.push(batch);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = full_.pop();

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }

  free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(Batch);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
