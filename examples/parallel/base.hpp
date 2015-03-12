#include <boost/thread.hpp>
#include <caffe/parallel.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/solver.hpp>
#include <glog/logging.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace caffe;

// Context for a solver running in a thread. Both initialization and run
// of the solver are done on the thread, to point to the same instance of the
// thread-local Caffe singleton.
class SolverContext : public Threaded {
 public:
  SolverContext(const Params<float>* params,
                const SolverParameter& solver_param, boost::barrier* barrier,
                int index, bool parallel, SGDSolver<float>* solver = NULL)
      : params_(params),
        solver_param_(solver_param),
        barrier_(barrier),
        index_(index),
        parallel_(parallel),
        worker_(solver == NULL),
        solver_(solver),
        syncs_() {

    // First solver does testing, display, snapshots etc., others
    // only training.
    if (worker_) {
      solver_param_.clear_display();
      solver_param_.clear_snapshot();
    }
  }

  inline const Params<float>* params() const {
    return params_;
  }
  inline SGDSolver<float>* solver() const {
    return solver_;
  }
  inline void add_sync(P2PSync<float>* value) {
    syncs_.push_back(value);
  }

  virtual void run() {
    if (worker_) {
      if (parallel_) {
        solver_ = new ParallelSolver<float>(solver_param_, false, barrier_,
                                            index_);
      } else {
        solver_ = new SGDSolver<float>(solver_param_);
      }
    }
    params_->configure(solver_);
    if (parallel_) {
      for (int i = 0; i < syncs_.size(); ++i) {
        ((ParallelSolver<float>*) solver_)->add_sync(syncs_[i]);
      }
    }
    solver_->Solve();
    // Wait until asked to stop before destroying, monitor might
    // still be accessing fields
    if (worker_)
      while (!must_stop())
        sleep(1);
    if (worker_)
      delete solver_;
  }

 protected:
  const Params<float>* params_;
  SolverParameter solver_param_;
  boost::barrier* barrier_;
  const int index_;
  const bool parallel_;
  const bool worker_;
  SGDSolver<float>* solver_;
  vector<P2PSync<float>*> syncs_;

DISABLE_COPY_AND_ASSIGN(SolverContext);
};

// Displays stats about a set of solvers. Also keeps track and updates
// the global count of iterations (needed to adjust hyperparams).
class Monitor : public Threaded {
 public:
  Monitor(const vector<SolverContext*>& solvers)
      : solvers_(solvers),
        total_iters_("total") {
  }

  virtual ~Monitor() {
  }

  void run() {
    int every_seconds = 10;
    while (!must_stop()) {
      sleep(every_seconds);
      ostringstream s;
      step(&s);
      s << "\n";
      LOG(INFO)<< s.str();
    }
  }

  int batch(SGDSolver<float>* solver) {
    for (int i = 0; i < solver->net()->layers().size(); ++i) {
      Layer<float>* layer = solver->net()->layers()[i].get();
      if(layer->layer_param().has_data_param()) {
        return layer->layer_param().data_param().batch_size();
      }
    }
    return 0;
  }

  void step(ostream* s) {
    *s << "Monitor - images: ";

    int total = 0;
    bool all = true;  // TODO remove
    for (int i = 0; i < solvers_.size(); ++i) {
      SolverContext* ctx = solvers_[i];
      int n = 0;
      if(ctx->solver()) {
        n = ctx->solver()->iter() * batch(ctx->solver());
      }
      total += n;
      *s << n << ", ";
      if (!n)
      all = false;
    }
    if (all) {
      //cudaProfilerStart();
      //LOG(INFO)<< "Started profiler\n";
    }
    total_iters_.value(total);
    total_iters_.show(*s);
    *s << ", ";
    stats(*s);
  }

  virtual void stats(ostream& s) const {
  }

protected:
  const vector<SolverContext*>& solvers_;
  Meter total_iters_;

  DISABLE_COPY_AND_ASSIGN(Monitor);
};
