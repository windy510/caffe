#include <boost/thread.hpp>
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

void Meter::show(std::ostream& s, bool all, int indent) const {
  if (all && children_.size()) {
    for (int c = 0; c < children_.size(); ++c) {
      for (int i = 0; i < indent + 2; ++i) {
        s << " ";
      }
      children_[c]->show(s, all, indent + 2);
      s << "\n";
    }
  } else {
    ptime now = microsec_clock::local_time();
    uint64_t value = this->value();
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
}

//

template<typename Dtype>
static size_t len(const vector<shared_ptr<Blob<Dtype> > >& params) {
  size_t len = 0;
  for (int i = 0; i < params.size(); ++i)
    len += params[i]->count();
  return len;
}

template<typename Dtype>
static bool file_map(const string& file, Dtype*& buffer, size_t len) {
  struct stat st_buf;
  bool exists = stat(file.c_str(), &st_buf) == 0;
  int fd = open(file.c_str(), O_RDWR | O_CREAT,  //
                S_IRWXU | S_IRWXG | S_IRWXO);
  CHECK(!ftruncate(fd, len * sizeof(Dtype)));
  buffer = (Dtype*) mmap(NULL,  //
      len * sizeof(Dtype),
      PROT_READ | PROT_WRITE,
      MAP_SHARED, fd, 0);
  close(fd);
  return exists;
}

enum Op {
  check,
  copy_cpu,
  copy_gpu,
  replace_cpu,
  replace_gpu
};

template<typename Dtype>
static void apply_buffers(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                          Dtype* buffer, size_t len_used, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->data()->size();
    switch (op) {
      case check:
        // NOP, just check sizes match
        break;
      case copy_cpu:
        // Init to current values of blobs
        memcpy(ptr, blobs[i]->data()->cpu_data(), size);
        break;
      case copy_gpu:
        // Init to current values of blobs
#ifndef CPU_ONLY
        CUDA_CHECK(cudaMemcpy(ptr, blobs[i]->data()->cpu_data(), size,  //
                              cudaMemcpyHostToDevice));
#endif
        break;
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
    }
    ptr += size / sizeof(Dtype);
    CHECK(ptr <= buffer + len_used);
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = buffer + check;
  CHECK_EQ(expect, ptr);
}

template<typename Dtype>
Params<Dtype>::Params(const SGDSolver<Dtype>& to_copy)
    : len_used_(len<Dtype>(to_copy.net()->params())),
      len_buff_(align(len_used_)),
      iterations_() {
}

template<typename Dtype>
size_t Params<Dtype>::align(const size_t len) {
  size_t m = len;
#ifndef CPU_ONLY
  m = max(m, P2PSync<Dtype>::chunks(len) * P2PSync<Dtype>::CHUNK);
#endif
  return m;
}

//

template<typename Dtype>
CPUParams<Dtype>::CPUParams(const SGDSolver<Dtype>& to_copy,
                            const string& file_map_dir)
    : Params<Dtype>(to_copy),
      mmap_(!file_map_dir.empty()) {

  // Allocate or map buffers
  bool data_mmap = false;
  bool hist_mmap = false;
  if (mmap_) {
    mkdir(file_map_dir.c_str(), 0777);
    data_mmap = file_map(file_map_dir + "/data", data_, len_buff_);
    hist_mmap = file_map(file_map_dir + "/hist", hist_, len_buff_);
  } else {
    CaffeMallocHost((void**) &data_, len_buff_ * sizeof(Dtype));
    CaffeMallocHost((void**) &hist_, len_buff_ * sizeof(Dtype));
    memset(data_, 0, len_buff_ * sizeof(Dtype));
    memset(hist_, 0, len_buff_ * sizeof(Dtype));
  }

  // Copy blob values if file maps do not already exist
  const vector<shared_ptr<Blob<Dtype> > >& net = to_copy.net()->params();
  const vector<shared_ptr<Blob<Dtype> > >& sol = to_copy.history();
  apply_buffers(net, data_, len_used_, data_mmap ? check : copy_cpu);
  apply_buffers(sol, hist_, len_used_, hist_mmap ? check : copy_cpu);
}

template<typename Dtype>
CPUParams<Dtype>::~CPUParams() {
  if (mmap_) {
    munmap((void*) data_, len_buff_ * sizeof(Dtype));
    munmap((void*) hist_, len_buff_ * sizeof(Dtype));
  } else {
    CaffeFreeHost((void*) data_);
    CaffeFreeHost((void*) hist_);
  }
}

template<typename Dtype>
void CPUParams<Dtype>::configure(SGDSolver<Dtype>* solver) const {
  const vector<shared_ptr<Blob<Dtype> > > &net = solver->net()->params();
  const vector<shared_ptr<Blob<Dtype> > > &sol = solver->history();
  apply_buffers(net, data_, len_used_, replace_cpu);
  apply_buffers(sol, hist_, len_used_, replace_cpu);
  solver->iter_total(&iterations_);
}

//

#ifndef CPU_ONLY
#include <cuda_runtime.h>

template<typename Dtype>
GPUParams<Dtype>::GPUParams(const SGDSolver<Dtype>& to_copy, int device)
    : Params<Dtype>(to_copy),
      device_(device) {

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  const size_t size = len_buff_ * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** ) &data_, size));
  CUDA_CHECK(cudaMalloc((void** ) &hist_, size));

  // Copy blob values
  const vector<shared_ptr<Blob<Dtype> > >& net = to_copy.net()->params();
  const vector<shared_ptr<Blob<Dtype> > >& sol = to_copy.history();
  apply_buffers(net, data_, len_used_, copy_gpu);
  apply_buffers(sol, hist_, len_used_, copy_gpu);

  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
  CUDA_CHECK(cudaFree((void* ) data_));
  CUDA_CHECK(cudaFree((void* ) hist_));
}

template<typename Dtype>
void GPUParams<Dtype>::configure(SGDSolver<Dtype>* solver) const {
  const vector<shared_ptr<Blob<Dtype> > >& net = solver->net()->params();
  const vector<shared_ptr<Blob<Dtype> > >& sol = solver->history();
  apply_buffers(net, data_, len_used_, replace_gpu);
  apply_buffers(sol, hist_, len_used_, replace_gpu);
  solver->iter_total(&iterations_);
}

template<typename Dtype>
void GPUParams<Dtype>::FileMapper::run() {
  Dtype* data;
  Dtype* hist;
  mkdir(file_map_dir_.c_str(), 0777);
  file_map(file_map_dir_ + "/data", data, params_.len_buff());
  file_map(file_map_dir_ + "/hist", hist, params_.len_buff());

  cudaStream_t stream;
  CUDA_CHECK(cudaSetDevice(params_.device()));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const size_t len = params_.len_buff() * sizeof(Dtype);

  try {
    while (!must_stop()) {
      usleep(100000);
      const cudaMemcpyKind d2h = cudaMemcpyDeviceToHost;
      CUDA_CHECK(cudaMemcpyAsync(data, params_.data(), len, d2h, stream));
      CUDA_CHECK(cudaMemcpyAsync(hist, params_.hist(), len, d2h, stream));
    }
    throw boost::thread_interrupted();
  } catch (...) {
    CUDA_CHECK(cudaStreamDestroy(stream));
    munmap((void*) data, len);
    munmap((void*) hist, len);
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(CPUParams);
#ifndef CPU_ONLY
INSTANTIATE_CLASS(GPUParams);
#endif

#endif
}
