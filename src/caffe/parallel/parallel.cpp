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
  time_point now = steady_clock::now();
  uint64_t value = value();
  double delta = value - last_;
  duration<double> secs = now - time_;
  double per_s = delta / secs;
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

template<typename Dtype>
static bool file_map(const string& file, size_t len, Dtype** buffer) {
  struct stat st_buf;
  bool exists = stat(file.c_str(), &st_buf) == 0;
  int fd = open(file.c_str(), O_RDWR | O_CREAT,  //
                S_IRWXU | S_IRWXG | S_IRWXO);
  CHECK(!ftruncate(fd, len * sizeof(Dtype)));
  *buffer = (Dtype*) mmap(NULL,  //
      len * sizeof(Dtype),
      PROT_READ | PROT_WRITE,
      MAP_SHARED, fd, 0);
  close(fd);
  return exists;
}

enum Op {
  check,
  copy,
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
      case Op::check:
        // NOP, just check sizes match
        break;
      case Op::copy:
        // Init to current values of blobs if file doesn't already exists
        memcpy(ptr, blobs[i]->data()->cpu_data(), size);
        break;
      case Op::replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case Op::replace_gpu:
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
Params<Dtype>::Params(SGDSolver<Dtype>* solver, const string& file_map_dir)
    : len_used_(len<Dtype>(solver->net()->params())),
      len_buff_(align(len_used_)) {

  // Allocate or map buffers
  bool data_exists = false;
  bool hist_exists = false;
  if (file_map_dir.empty()) {
    CaffeMallocHost((void**) &data_, len_buff_ * sizeof(Dtype));
    CaffeMallocHost((void**) &hist_, len_buff_ * sizeof(Dtype));
    memset(data_, 0, len_buff_ * sizeof(Dtype));
    memset(hist_, 0, len_buff_ * sizeof(Dtype));
  } else {
    data_exists = file_map(file_map_dir + '/data', data_, len_buff_);
    hist_exists = file_map(file_map_dir + '/hist', hist_, len_buff_);
  }

  // Copy blob values if file maps do not already exist
  vector<shared_ptr<Blob<Dtype> > >& net = solver->net()->params();
  apply_buffers(net, data_, len_used_, data_exists ? Op::check : Op::copy);
  vector<shared_ptr<Blob<Dtype> > >& sol = solver->history();
  apply_buffers(sol, hist_, len_used_, hist_exists ? Op::check : Op::copy);

  iterations_ = 0;
}

template<typename Dtype>
Params<Dtype>::~Params() {
  CaffeFreeHost((void*) data_);
  CaffeFreeHost((void*) hist_);
}

template<typename Dtype>
void Params<Dtype>::configure(Solver<Dtype>* solver) const {
  vector<shared_ptr<Blob<Dtype> > >& net = solver->net()->params();
  apply_buffers(net, data_, len_used_, Op::replace_cpu);

  vector<shared_ptr<Blob<Dtype> > >& sol = solver->history();
  apply_buffers(sol, hist_, len_used_, Op::replace_cpu);

  solver->iter_total(&iterations_);
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

#ifndef CPU_ONLY
#include <cuda_runtime.h>

template<typename Dtype>
GPUParams<Dtype>::GPUParams(const Params<Dtype>& params, int device)
    : params_(params),
      device_(device) {

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  CUDA_CHECK(cudaSetDevice(device));
  const size_t size = params.len_buff() * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** ) &data_, size));
  CUDA_CHECK(cudaMalloc((void** ) &hist_, size));
  CUDA_CHECK(cudaMemcpy(data_, params.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hist_, params.hist(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(initial_device));
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
  CUDA_CHECK(cudaFree((void* ) data_));
  CUDA_CHECK(cudaFree((void* ) hist_));
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  vector<shared_ptr<Blob<Dtype> > >& net = solver->net()->params();
  apply_buffers(net, data_, params_.len_used(), Op::replace_gpu);

  vector<shared_ptr<Blob<Dtype> > >& sol = solver->history();
  apply_buffers(sol, hist_, params_.len_used(), Op::replace_gpu);

  solver->iter_total(&params_.iterations_);
}

INSTANTIATE_CLASS(Params);
#ifndef CPU_ONLY
INSTANTIATE_CLASS(GPUParams);
#endif

#endif
}
