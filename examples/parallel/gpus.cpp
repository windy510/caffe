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

// Trains a net on multiple GPUs on one box.
//
// Example launch on GPU 0 and 1:
// make -j
// export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
// export GLOG_logtostderr=1
// build/examples/parallel/gpus.bin examples/parallel/mnist_solver.prototxt 0:1

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc != 3) {
    printf("Usage: gpus.bin solver_proto_file gpu_id[:gpu_id][...]\n");
    return 1;
  }

  SolverParameter proto;
  ReadProtoFromTextFile(argv[1], &proto);

  vector<int> gpus;
  {
    vector<string> gpu_strings;
    boost::split(gpu_strings, argv[2], boost::is_any_of(":"));
    for (int i = 0; i < gpu_strings.size(); ++i)
      gpus.push_back(atoi(gpu_strings[i].c_str()));
  }

  // Barrier for solvers to wait on each other at start and shutdown
  boost::barrier barrier(gpus.size());

  // Create first solver
  proto.set_device_id(gpus[0]);
  ParallelSolver<float> first(proto, true, &barrier, 0);
  //  first.Restore("examples/parallel/lenet_iter_1000.solverstate");

  // Allocate params for each device, copying first solver weights
  vector<GPUParams<float>*> params(gpus.size());
  for (int i = 0; i < gpus.size(); ++i) {
    params[i] = new GPUParams<float>(first, gpus[i]);
  }

  // Create other solvers
  vector<SolverContext*> solvers(gpus.size());
  solvers[0] = new SolverContext(params[gpus[0]], proto, &barrier, 0, true,
                                 &first);
  for (int i = 1; i < gpus.size(); ++i) {
    int device = gpus[i];
    proto.set_device_id(device);
    solvers[i] = new SolverContext(params[device], proto, &barrier, i, true);
  }

  // Create synchronization
  P2PSync<float> sync(params);
  for (int i = 0; i < solvers.size(); ++i) {
    solvers[i]->add_sync(&sync);
  }

  // Monitor
  Monitor monitor(solvers);
  monitor.start();

  for (int i = 1; i < gpus.size(); ++i) {
    solvers[i]->start();
  }
  // Run first on current thread
  solvers[0]->run();

  monitor.stop();
  LOG(INFO)<< "Monitor stopped\n";

  for (int i = 1; i < solvers.size(); ++i)
    solvers[i]->stop();
  LOG(INFO)<< "Solvers stopped\n";
  for (int i = 0; i < solvers.size(); ++i)
    delete solvers[i];
  for (int i = 0; i < params.size(); ++i) {
    delete params[i];
  }
}

#else
int main(int argc, char *argv[]) {
}
#endif

