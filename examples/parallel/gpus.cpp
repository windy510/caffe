#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/net.hpp>
#include <caffe/parallel.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/solver.hpp>
#include <caffe/util/io.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "base.hpp"

using boost::lexical_cast;
using namespace caffe;
using namespace std;

#ifndef CPU_ONLY
#include <cuda_runtime.h>

// Trains a net on multiple GPUs on one box. Synchronization occurs between
// groups of GPUs, that need to be configured to fit the machine topology.
// E.g. two GPUs on the same PCI switch, as in the case of a K80 board, should
// be placed in a dedicated group.
//
// Example launch on GPU 0 and 1:
// make -j
// export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
// export GLOG_logtostderr=1
// build/examples/parallel/gpus.bin examples/parallel/mnist_solver.prototxt 0:1
//
// Example with three K80, each with ids 0-1, 2-3, and 4-5 (6 GPUs total)
// gpus.bin examples/parallel/mnist_solver.prototxt 0:1 2:3 4:5 0:2:4

// Monitors solvers and network
class GPUMonitor : public Monitor {
 public:
  GPUMonitor(const vector<SolverContext*>& solvers, double coherence,
             const vector<P2PSync<float>*> syncs)
      : Monitor(solvers, coherence),
        syncs_(syncs) {
  }

  void stats(ostream& s) const {
    if (DETAILLED) {
      s << "\n";
    }
    for (int sync = 0; sync < syncs_.size(); ++sync) {
      s << "sync " << sync << "\n";
      for (int g = 0; syncs_[sync] && g < syncs_[sync]->gpus().size(); ++g) {
        P2PSync<float>::GPU* gpu = syncs_[sync]->gpus()[g].get();
        s << "  gpu " << gpu->params()[gpu->index()]->device() << "\n";
        gpu->sent().show(s, DETAILLED, 2);
        if (!DETAILLED) {
          s << ", ";
        }
        gpu->recv().show(s, DETAILLED, 2);
      }
    }
  }

  const static bool DETAILLED = true;
  const vector<P2PSync<float>*> syncs_;
};

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc < 3) {
    printf("Usage: gpus.bin solver_proto_file "
           "gpu_id[:gpu_id][...] gpu_id[:gpu_id][...] ...\n");
    return 1;
  }

  SolverParameter proto;
  ReadProtoFromTextFile(argv[1], &proto);

  // Parse gpu ids and groups
  vector<int> gpus;
  vector<vector<int> > groups;
  for (int g = 2; g < argc; ++g) {
    vector<string> ids;
    boost::split(ids, argv[g], boost::is_any_of(":"));
    vector<int> group;
    for (int i = 0; i < ids.size(); ++i) {
      int gpu = atoi(ids[i].c_str());
      bool in = (std::find(gpus.begin(), gpus.end(), gpu) != gpus.end());
      if (!in) {
        gpus.push_back(gpu);
      }
      group.push_back(gpu);
    }
    groups.push_back(group);
  }

  // Create first solver
  proto.set_device_id(gpus[0]);
  SGDSolver<float> first(proto);
//  first.Restore("examples/parallel/lenet_iter_1000.solverstate");

  // Device to params map
  map<int, GPUParams<float>*> params;
  vector<GPUParams<float>::FileMapper*> debug(gpus.size());

  // Allocate params for each device, copying first solver weights
  for (int i = 0; i < gpus.size(); ++i) {
    params[gpus[i]] = new GPUParams<float>(first, gpus[i]);
    debug[i] = new GPUParams<float>::FileMapper(
        *(params[gpus[i]]), "/dev/shm/gpu" + lexical_cast<string>(i));
    debug[i]->start();
  }

  // Create other solvers
  vector<SolverContext*> solvers(gpus.size());
  solvers[0] = new SolverContext(params[gpus[0]], proto, &first);
  for (int i = 1; i < gpus.size(); ++i) {
    int device = gpus[i];
    proto.set_device_id(device);
    solvers[i] = new SolverContext(params[device], proto);
    solvers[i]->start();
  }

  // Create synchronizations
  vector<P2PSync<float>*> syncs(groups.size());
  vector<vector<GPUParams<float>*> > params_groups(groups.size());
  for (int group = 0; group < syncs.size(); ++group) {
    if (groups[group].size() > 1) {
      for (int gpu = 0; gpu < groups[group].size(); ++gpu) {
        int device = groups[group][gpu];
        params_groups[group].push_back(params[device]);
      }
      syncs[group] = new P2PSync<float>(params_groups[group]);
      syncs[group]->start();
    }
  }

  // Start monitor
  GPUMonitor monitor(solvers, .8, syncs);
  monitor.start();

  // Run first on current thread
  solvers[0]->run();

  monitor.stop();
  LOG(INFO)<< "Monitor stop\n";

  for (int i = 0; syncs[i] && i < syncs.size(); ++i) {
    syncs[i]->stop();
    delete syncs[i];
  }
  for (int i = 1; i < solvers.size(); ++i)
    solvers[i]->stop();
  for (int i = 0; i < solvers.size(); ++i)
    delete solvers[i];
  for (int i = 0; i < gpus.size(); ++i) {
    debug[i]->stop();
    delete debug[i];
    delete params[gpus[i]];
  }
}

#else
int main(int argc, char *argv[]) {
}
#endif

