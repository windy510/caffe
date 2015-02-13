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
                const SolverParameter& solver_param, SGDSolver<float>* solver =
                NULL)
      : params_(params),
        solver_param_(solver_param),
        worker_(solver == NULL),
        solver_(solver) {

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
  inline Solver<float>* solver() const {
    return solver_;
  }

  virtual void run() {
    if (worker_) {
      solver_ = new SGDSolver<float>(solver_param_, true);
      CHECK(!solver_->test_nets().size());  // No testing
    }
    params_->configure(solver_);
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
  const bool worker_;
  SGDSolver<float>* solver_;

DISABLE_COPY_AND_ASSIGN(SolverContext);
};

// Displays stats about a set of solvers. Also keeps track and updates
// the global count of iterations (needed to adjust hyperparams).
class Monitor : public Threaded {
 public:
  // Coherence is an attempt to express how total iterations should be counted.
  // If all solvers are perfectly in sync, the network could be seen as having
  // trained the sum of each solver iterations (coherence 1). If
  // synchronization has no effect, they are training independently, and
  // iterations is total / solvers (coherence 0). Depending on the ratio of
  // compute v.s. training, we guess numbers between .5 and .8 should be tried.
  Monitor(const vector<SolverContext*>& solvers, double coherence)
      : solvers_(solvers),
        coherence_(coherence),
        total_iters_("total") {
  }

  virtual ~Monitor() {
  }

  void run() {
    int every_seconds = 10;
    time_t start = time(0);
    while (!must_stop()) {
      sleep(every_seconds);
      ostringstream s;
      step(&s);
      s << "\n";
      LOG(INFO)<< s.str();
      LOG(INFO)<< "Training time: " << (time(0) - start);
    }
  }

  void step(ostream* s) {
    *s << "Monitor - iters: ";

    int total = 0;
    bool all = true;  // TODO remove
    for (int i = 0; i < solvers_.size(); ++i) {
      SolverContext* ctx = solvers_[i];
      int n = ctx->solver() ? ctx->solver()->iter() : 0;
      total += n;
      *s << n << ", ";
      if (!n)
      all = false;
    }
    for (int i = 0; i < solvers_.size(); ++i) {
      double c = coherence_;
      double n = total * c + (total / solvers_.size()) * (1 - c);
      solvers_[i]->params()->iterations(n);
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
  const double coherence_;
  Meter total_iters_;

  DISABLE_COPY_AND_ASSIGN(Monitor);
};
