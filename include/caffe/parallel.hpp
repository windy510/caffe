#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <boost/atomic/atomic.hpp>
#include <boost/chrono.hpp>
#include <sstream>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/internal_thread.hpp"

using boost::chrono::time_point;
using boost::chrono::steady_clock;
using boost::chrono::duration;

// The following classes enable data-parallel training, over multiple CPU
// cores, GPUs, and machines. It uses asynchronous SGD to independently max out
// networking and compute resources.

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
        time_(steady_clock::now()) {
  }

  inline uint64_t value() const {
    return value_.load(boost::memory_order_relaxed);
  }
  inline void value(uint64_t value) {
    value_.store(value);
  }
  inline void operator++(int) {
    value_.fetch_add(1, boost::memory_order_relaxed);
  }

  void show(std::ostream& s) const;

 protected:
  const string name_;
  const uint64_t unit_size_;
  mutable boost::atomic<uint64_t> value_;
  mutable uint64_t last_;
  mutable time_point time_;

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
  // Allocate buffers compatible with the given solver, optionally mapped to
  // file (e.g. in /dev/shm) for multi-process configurations or debugging.
  Params(SGDSolver<Dtype>* solver, const string& file_map_dir = "");
  virtual ~Params();

  inline size_t len_used() const {
    return len_used_;
  }
  inline size_t len_buff() const {
    return len_buff_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* hist() const {
    return hist_;
  }
  inline int iterations() {
    return iterations_;
  }
  inline void iterations(int value) {
    iterations_ = value;
  }

  // Replaces solvers parameters by Params's buffers.
  void configure(Solver<Dtype>* solver) const;

  // Align arrays to all potential chunk sizes to avoid boundary checks
  static size_t align(const size_t len);

 protected:
  const size_t len_used_;       // Actually used
  const size_t len_buff_;       // Allocated aligned to potential chunks
  Dtype* data_;
  Dtype* hist_;
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
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* hist() const {
    return hist_;
  }

 protected:
  const Params<Dtype>& params_;
  const int device_;
  Dtype* data_;
  Dtype* hist_;

DISABLE_COPY_AND_ASSIGN(GPUParams);
};

//

// Syncs params between GPUs on single box.
template<typename Dtype>
class P2PSync {
 public:
  P2PSync(const vector<GPUParams<Dtype>*> params);
  virtual ~P2PSync();

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
  class P2PEmit : public Threaded {
   public:
    P2PEmit(const P2PSync& sync, const vector<GPUParams<Dtype>*> params,
            int index);
    virtual ~P2PEmit();

    void run();

    const P2PSync& sync_;
    const vector<GPUParams<Dtype>*> params_;
    const int index_;
    const uint32_t chunks_;

  DISABLE_COPY_AND_ASSIGN(P2PEmit);
  };

  vector<shared_ptr<P2PEmit>> emits_;

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
