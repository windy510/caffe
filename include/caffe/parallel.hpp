#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <sstream>

#include <boost/dynamic_bitset.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;
using std::deque;

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
 Linux CUDA 7.
 */
namespace boost {
class barrier;
}

namespace caffe {

// Helper to write components running in their own threads
class Threaded : protected InternalThread {
 public:
  Threaded()
      : InternalThread() {
  }

  virtual void start() {
    this->StartInternalThread();
  }
  virtual void stop() {
    this->StopInternalThread();
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
  Meter(const string& name, double unit_size = 0)
      : name_(name),
        unit_size_(unit_size),
        value_(),
        last_(),
        time_(microsec_clock::local_time()) {
  }

  inline double value() const {
    return value_;
  }
  inline void value(double value) {
    value_ = value;
  }
  inline void tick() {
    add(1);
  }
  inline void add(double value) {
    value_ += value;
  }

  void show(std::ostream& s) const;

 protected:
  const string name_;
  const double unit_size_;
  mutable double value_, last_;
  mutable ptime time_;  // Switch to Boost.Chrono when default on 12.04

DISABLE_COPY_AND_ASSIGN(Meter);
};

//

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array, that the buffers
// are sufficiently long for chuncking alignments, and potentially other future
// requirements.
template<typename Dtype>
class Params {
 public:
  Params(const SGDSolver<Dtype>& to_copy);
  virtual ~Params() {
  }

  inline size_t len_used() const {
    return length_;
  }
  inline size_t len_buff() const {
    return length_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }
  inline Dtype* hist() const {
    return hist_;
  }

  // Replaces solvers parameters by Params's buffers.
  virtual void configure(SGDSolver<Dtype>* solver) const = 0;

 protected:
  const size_t length_;         // Length of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient
  Dtype* hist_;                 // Solver momentum

DISABLE_COPY_AND_ASSIGN(Params);
};

template<typename Dtype>
class CPUParams : public Params<Dtype> {
 public:
  // Allocate buffers compatible with the given solver, optionally mapped to
  // file (e.g. in /dev/shm) for visualization or debugging.
  CPUParams(const SGDSolver<Dtype>& to_copy,  //
      const string& file_map_dir = "");
  virtual ~CPUParams();

  virtual void configure(SGDSolver<Dtype>* solver) const;

 protected:
  const bool mmap_;

  using Params<Dtype>::length_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::hist_;
};

#ifndef CPU_ONLY

template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(const SGDSolver<Dtype>& to_copy, int device);
  virtual ~GPUParams();

  virtual void configure(SGDSolver<Dtype>* solver) const;

  inline int device() const {
    return device_;
  }

  // Loosely keeps a file maps in sync with GPU memory.
  // Same purpose as CPUParams one.
  class FileMapper : public Threaded {
   public:
    FileMapper(const GPUParams<Dtype>& params, const string& file_map_dir)
        : params_(params),
          file_map_dir_(file_map_dir) {
    }
    void run();

   protected:
    const GPUParams<Dtype>& params_;
    const string file_map_dir_;

  DISABLE_COPY_AND_ASSIGN(FileMapper);
  };

 protected:
  const int device_;

  using Params<Dtype>::length_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using Params<Dtype>::hist_;
};

//

template<typename Dtype>
class ParallelSolver;

// Syncs params between GPUs on single box.
template<typename Dtype>
class P2PSync {
 public:
  P2PSync(const vector<GPUParams<Dtype>*>& params);
  virtual ~P2PSync() {
  }

  inline const vector<GPUParams<Dtype>*> params() const {
    return params_;
  }

  class GPU;

  inline const vector<shared_ptr<GPU> >& gpus() const {
    return gpus_;
  }

 public:
  // Sends and receives gradients from one GPU to a group of others
  class GPU {
   public:
    GPU(const P2PSync& sync, int index);
    virtual ~GPU();

    inline const P2PSync& sync() const {
      return sync_;
    }
    inline const GPUParams<Dtype>& params() const {
      return params_;
    }
    inline int index() const {
      return index_;
    }

   protected:
    void send();

    const P2PSync& sync_;
    const GPUParams<Dtype>& params_;
    const int index_;
    vector<Dtype*> targets_;
    cudaStream_t stream_;

    template<typename U>
    friend class P2PSync;

  DISABLE_COPY_AND_ASSIGN(GPU);
  };

 protected:
  void send(int index) const;
  void sum_targets();

  const vector<GPUParams<Dtype>*> params_;
  vector<shared_ptr<GPU> > gpus_;

  template<typename U>
  friend class ParallelSolver;
  friend class GPU;

DISABLE_COPY_AND_ASSIGN(P2PSync);
};

#endif

//

template<typename Dtype>
class ParallelSolver : public SGDSolver<Dtype> {
 public:
  explicit ParallelSolver(const SolverParameter& param, bool init_test_nets,
                          boost::barrier* barrier, int index)
      : SGDSolver<Dtype>(param, false),
        barrier_(barrier),
        index_(index),
        syncs_() {
    this->Init();
    this->InitTrainNet();
    if (init_test_nets) {
      this->InitTestNets();
    }
    this->PreSolve();
    LOG(INFO)<<"Solver scaffolding done.";
  }
  virtual ~ParallelSolver() {
  }

  inline const vector<P2PSync<Dtype>*>& syncs() const {
    return syncs_;
  }
  inline void add_sync(P2PSync<Dtype>* value) {
    syncs_.push_back(value);
  }

protected:
  virtual void Iteration();

  boost::barrier* barrier_;
  const int index_;
  vector<P2PSync<Dtype>*> syncs_;

  DISABLE_COPY_AND_ASSIGN(ParallelSolver);
};

}

#endif
