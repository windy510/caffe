#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <sstream>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#include <caffe/caffe.hpp>
#include "caffe/parallel.hpp"

using namespace std;

namespace caffe {

void Meter::show(std::ostream& s) const {
  ptime now = microsec_clock::local_time();
  uint64_t value = value_;
  uint64_t delta = value - last_;
  uint64_t u_sec = (now - time_).total_microseconds();
  double per_s = delta * 1e6 / (u_sec ? u_sec : 1);
  last_ = value;
  time_ = now;
  s << name_ << " " << value << " (";
  if (unit_size_)
    s << (int) (per_s * unit_size_ / (1024 * 1024)) << " mb";
  else
    s << std::setprecision(2) << per_s;
  s << "/s)";
}

//

template<typename Dtype>
static size_t len(const vector<shared_ptr<Blob<Dtype> > >& params) {
  size_t len = 0;
  for (int i = 0; i < params.size(); ++i)
    len += params[i]->count();
  return len;
}

// Align arrays to all potential chunk sizes to avoid boundary checks
template<typename Dtype>
static size_t align(const size_t len) {
  size_t m = len;
#ifndef CPU_ONLY
  m = max(m, P2PSync<Dtype>::chunks(len) * P2PSync<Dtype>::CHUNK);
#endif
  return m;
}

template<typename Dtype>
Params<Dtype>::Params(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                      const string& file_map)
    : len_used_(len<Dtype>(blobs)),
      len_buff_(align<Dtype>(len_used_)) {

  bool exists = false;
  if (file_map.empty()) {
    CaffeMallocHost((void**) &cpu_, len_buff_ * sizeof(Dtype));
    memset(cpu_, 0, len_buff_ * sizeof(Dtype));
  } else {
    struct stat st_buf;
    exists = stat(file_map.c_str(), &st_buf) == 0;
    int fd = open(file_map.c_str(), O_RDWR | O_CREAT,  //
                  S_IRWXU | S_IRWXG | S_IRWXO);
    CHECK(!ftruncate(fd, len_buff_ * sizeof(Dtype)));
    cpu_ = (Dtype*) mmap(NULL,  //
        len_buff_ * sizeof(Dtype),
        PROT_READ | PROT_WRITE,
        MAP_SHARED, fd, 0);
    close(fd);
  }

  Dtype* cpu = cpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->data()->size();
    // Init to current values of blobs if file doesn't already exists
    if (!exists)
      memcpy(cpu, blobs[i]->data()->cpu_data(), size);
    cpu += size / sizeof(Dtype);
    CHECK(cpu <= cpu_ + len_used_);
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = cpu_ + check;
  CHECK_EQ(expect, cpu);

  iterations_ = 0;
}

template<typename Dtype>
Params<Dtype>::~Params() {
  CaffeFreeHost((void*) cpu_);
}

template<typename Dtype>
void Params<Dtype>::configure(Solver<Dtype>* solver) const {
  // Replace weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver->net()->params();
  Dtype* cpu = cpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_cpu_data(cpu);
    cpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(cpu <= cpu_ + len_used_);
  }
  // Check sizes
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = cpu_ + check;
  CHECK_EQ(expect, cpu);

  solver->iter_total(&iterations_);
}

//

#ifndef CPU_ONLY
#include <cuda_runtime.h>

template<typename Dtype>
GPUParams<Dtype>::GPUParams(const Params<Dtype>& params, int device)
    : params_(params),
      device_(device) {

  int current;
  CUDA_CHECK(cudaGetDevice(&current));
  CUDA_CHECK(cudaSetDevice(device));
  const size_t size = params.len_buff() * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** ) &gpu_, size));
  CUDA_CHECK(cudaMemcpy(gpu_, params.cpu(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(current));
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
  CUDA_CHECK(cudaFree((void* ) gpu_));
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  // Replace GPU weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver->net()->params();
  Dtype* gpu = gpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_gpu_data(gpu);
    gpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(gpu <= gpu_ + params_.len_used());
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = gpu_ + check;
  CHECK_EQ(expect, gpu);

  solver->iter_total(&params_.iterations_);
}

//

template<typename Dtype>
GPUStream<Dtype>::GPUStream() {
// TODO bench priorities
//  int least, greatest;
//  cudaDeviceGetStreamPriorityRange(&least, &greatest);
//  cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, least);
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
}

template<typename Dtype>
GPUStream<Dtype>::~GPUStream() {
  cudaStreamDestroy(stream_);
}

INSTANTIATE_CLASS(Params);
#ifndef CPU_ONLY
INSTANTIATE_CLASS(GPUParams);
#endif

#endif
}
