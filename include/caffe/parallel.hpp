#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <sstream>

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

// The following classes enable data-parallel training, over multiple CPU
// cores, GPUs, and machines. It uses asynchronous SGD to independently max out
// networking and compute resources.

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
  virtual void stop(bool wait = false) {
    if (wait) {
      this->StopInternalThread();
    } else {
      this->RequestInternalThreadStop();
    }
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
        unit_size_(unit_size),
        children_(),
        value_(),
        last_(),
        time_(microsec_clock::local_time()) {
  }

  inline uint64_t value() const {
    uint64_t value = 0;
    if (children_.size()) {
      for (int i = 0; i < children_.size(); ++i) {
        value += children_[i]->value();
      }
    } else {
      value = value_;
    }
    return value;
  }
  inline void value(uint64_t value) {
    value_ = value;
  }
  inline void tick() {
    value_++;
  }
  void add_child(const Meter* meter) {
    children_.push_back(meter);
  }

  void show(std::ostream& s, bool all = false, int indent = 0) const;

 protected:
  const string name_;
  const uint64_t unit_size_;
  vector<const Meter*> children_;
  mutable uint64_t value_, last_;
  mutable ptime time_;  // Switch to Boost.Chrono when default on 12.04

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
  Params(const SGDSolver<Dtype>& to_copy);
  virtual ~Params() {
  }

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
  inline int iterations() const {
    return iterations_;
  }
  inline void iterations(int value) const {
    iterations_ = value;
  }

  // Replaces solvers parameters by Params's buffers.
  virtual void configure(SGDSolver<Dtype>* solver) const = 0;

  // Align arrays to all potential chunk sizes to avoid boundary checks
  static size_t align(const size_t len);

 protected:
  const size_t len_used_;       // Actually used
  const size_t len_buff_;       // Allocated aligned to potential chunks
  Dtype* data_;                 // Network parameters
  Dtype* hist_;                 // Solver momentum
  mutable int iterations_;      // Total iterations across solvers

DISABLE_COPY_AND_ASSIGN(Params);
};

template<typename Dtype>
class CPUParams : public Params<Dtype> {
 public:
  // Allocate buffers compatible with the given solver, optionally mapped to
  // file (e.g. in /dev/shm) for multi-process configurations or debugging.
  CPUParams(const SGDSolver<Dtype>& to_copy, const string& file_map_dir = "");
  virtual ~CPUParams();

  virtual void configure(SGDSolver<Dtype>* solver) const;

 protected:
  const bool mmap_;

  using Params<Dtype>::len_used_;
  using Params<Dtype>::len_buff_;
  using Params<Dtype>::data_;
  using Params<Dtype>::hist_;
  using Params<Dtype>::iterations_;
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

  // Loosely keeps a file map of the weights in sync with GPU memory.
  // Same purpose as CPUParams one.
  class FileMapper : public Threaded {
   public:
    FileMapper(const GPUParams<float>& params, const string& file_map_dir)
        : params_(params),
          file_map_dir_(file_map_dir) {
    }
    void run();

   protected:
    const GPUParams<float>& params_;
    const string file_map_dir_;

  DISABLE_COPY_AND_ASSIGN(FileMapper);
  };

 protected:
  const int device_;

  using Params<Dtype>::len_used_;
  using Params<Dtype>::len_buff_;
  using Params<Dtype>::data_;
  using Params<Dtype>::hist_;
  using Params<Dtype>::iterations_;
};

//

// Syncs params between GPUs on single box.
template<typename Dtype>
class P2PSync {
 public:
  P2PSync(const vector<GPUParams<Dtype>*>& params);
  virtual ~P2PSync() {
  }

  void start();
  void stop();

  class GPU;

  const vector<shared_ptr<GPU> >& gpus() const {
    return gpus_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

  // TODO bench, auto tune?
  static const int CHUNK = 262144;
  static const int QUEUE = 4;

 protected:
  static vector<shared_ptr<GPU> > create_gpus(
      const vector<GPUParams<Dtype>*>& params);

  class Callback;

  // Emulates P2P multicast by sending from one GPU to others in group
  class Multicast {
   public:
    Multicast(const GPU& gpu);
    ~Multicast();

    const static int LENGTH = CHUNK * sizeof(Dtype);
    const int index_;
    Dtype* source_;
    vector<Dtype*> targets_;
    vector<Callback> callbacks_;
    uint32_t pending_targets_;
    uint32_t chunk_;

  DISABLE_COPY_AND_ASSIGN(Multicast);
  };

  class Callback {
   public:
    Callback()
        : queue_(NULL),
          multicast_(NULL) {
    }
    blocking_queue<Multicast*>* queue_;
    Multicast* multicast_;
  };

 public:
  // Sends and receives gradients from one GPU to a group of others
  class GPU : public Threaded {
   public:
    GPU(const P2PSync& sync, const vector<GPUParams<Dtype>*> params, int index);
    virtual ~GPU();

    void run();

    static void CUDART_CB add_to_queue(cudaStream_t stream, cudaError_t status,
                                       void* data);

    inline const P2PSync& sync() const {
      return sync_;
    }
    inline const vector<GPUParams<Dtype>*>& params() const {
      return params_;
    }
    inline int index() const {
      return index_;
    }
    inline const Meter& sent() const {
      return sent_;
    }
    inline const Meter& cycles() {
      return cycles_;
    }

   protected:
    const P2PSync& sync_;
    const vector<GPUParams<Dtype>*> params_;
    const int index_;
    const uint32_t chunks_;

    // TODO switch to spsc_queue (Boost 1.53 not packaged on Ubuntu 12.04)
    blocking_queue<Multicast*> queue_;

    // Perf counters
    Meter sent_;
    vector<shared_ptr<Meter> > sent_to_each_;
    Meter cycles_;

    template<typename U>
    friend class P2PSync;

  DISABLE_COPY_AND_ASSIGN(GPU);
  };

 protected:
  const shared_ptr<boost::barrier> barrier_;
  vector<shared_ptr<GPU> > gpus_;

  // Perf counters
  Meter sent_;
  Meter cycles_;

DISABLE_COPY_AND_ASSIGN(P2PSync);
};

template<typename Dtype>
void p2p_sync_send(Dtype* data, Dtype* copy, size_t off, Dtype* chunk,
                   const cudaStream_t& stream);
template<typename Dtype>
void p2p_sync_recv(Dtype* data, Dtype* copy, size_t off, Dtype* chunk,
                   const cudaStream_t& stream);

#endif

}

#endif
