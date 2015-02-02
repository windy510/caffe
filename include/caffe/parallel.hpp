#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <boost/date_time/posix_time/posix_time.hpp>
#include <sstream>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/internal_thread.hpp"

using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;

// The following classes enable data-parallel training, over multiple CPU cores,
// GPUs, and machines. It uses a form of asynchronous SGD that can independently
// max out both networking and compute resources.

namespace caffe {

// Helper to write components running in their own threads
class Threaded : public InternalThread {
 public:
  Threaded()
      : InternalThread() {
  }

  virtual void start() {
    this->StartInternalThread();
  }
  virtual void stop() {
    this->WaitForInternalThreadToExit();
  }

  virtual void run() = 0;

 protected:
  void InternalThreadEntry() {
    run();
  }

DISABLE_COPY_AND_ASSIGN(Threaded);
};

// Helper for perf metrics
class Meter {
 public:
  // If unit_size is specified, meter will display bandwidth as size * count/s
  Meter(const string& name, uint64_t unit_size = 0)
      : name_(name),
        unit_size_(unit_size),  //
        value_(),
        last_(),
        time_(microsec_clock::local_time()) {
  }

  inline uint64_t value() const {
    return value_;
  }
  inline void value(uint64_t value) {
    value_ = value;
  }
  inline void operator++(int) {
    value_++;
  }

  void show(std::ostream& s) const;

 protected:
  const string name_;
  const uint64_t unit_size_;
  mutable uint64_t value_, last_;
  mutable ptime time_;  // TODO find a monotonic clock

DISABLE_COPY_AND_ASSIGN(Meter);
};

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array, that the buffers
// are sufficiently long for chuncking alignments, and potentially other future
// requirements. Also keeps track of the total iterations on those weights, to
// get correct hyper-parameters schedules across multiple solvers.
// TODO keep track of total iterations also between machines.
template<typename Dtype>
class Params {
 public:
  // Allocate a buffer compatible with the given blobs, optionally mapped to a
  // file (/dev/shm) for multi-process configurations or debugging.
  Params(const vector<shared_ptr<Blob<Dtype> > >& blobs,  //
      const string& file_map = "");
  virtual ~Params();

  inline size_t len_used() const {
    return len_used_;
  }
  inline size_t len_buff() const {
    return len_buff_;
  }
  inline Dtype* cpu() const {
    return cpu_;
  }
  inline int iterations() {
    return iterations_;
  }
  inline void iterations(int value) {
    iterations_ = value;
  }

  // Replaces solvers parameters by the shared buffer. Solvers then run on
  // the same weights without synchronization (Hogwild). See hogwild.cpp in
  // /examples for details and BLAS requirements.
  void configure(Solver<Dtype>* solver) const;

 protected:
  const size_t len_used_;       // Actually used
  const size_t len_buff_;       // Allocated aligned to potential chunks
  Dtype* cpu_;
  mutable int iterations_;      // Total iterations across solvers

  template<typename U>
  friend class GPUParams;

DISABLE_COPY_AND_ASSIGN(Params);
};

#ifndef CPU_ONLY

// Params on a GPU
template<typename Dtype>
class GPUParams {
 public:
  GPUParams(const Params<Dtype>& params, int device);
  virtual ~GPUParams();
  void configure(Solver<Dtype>* solver) const;

  inline const Params<Dtype>& params() const {
    return params_;
  }
  inline int device() const {
    return device_;
  }
  inline Dtype* gpu() const {
    return gpu_;
  }

 protected:
  const Params<Dtype>& params_;
  const int device_;
  Dtype* gpu_;

DISABLE_COPY_AND_ASSIGN(GPUParams);
};

template<typename Dtype>
class GPUStream {
 public:
  GPUStream();
  virtual ~GPUStream();

  const cudaStream_t& stream() const {
    return stream_;
  }

 protected:
  cudaStream_t stream_;

DISABLE_COPY_AND_ASSIGN(GPUStream);
};

// Syncs params between host and GPU memory.
template<typename Dtype>
class P2PSync : public Threaded {
 public:
  P2PSync(const GPUParams<Dtype>& params);

  virtual ~P2PSync();

  void run();

  inline const Meter& calls() const {
    return calls_;
  }
  inline const Meter& cycles() {
    return cycles_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

  // TODO bench, auto tune?
  static const int CHUNK = 262144;

 protected:
  void push(uint32_t chunk);

  const uint32_t chunks_;
  const GPUParams<Dtype>& params_;
  Dtype* gpu_last_;

  // Perf counters
  Meter calls_, cycles_;

DISABLE_COPY_AND_ASSIGN(P2PSync);
};

template<typename Dtype>
void p2p_sync_kernel(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off,
                     const cudaStream_t& stream, size_t chunk);

#endif

}

#endif
