#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

// TODO single solver until multi-gpu merge
static const int solver_count = 1;

DataReader::DataReader(const LayerParameter& param)
    : queues_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->reader_queues_.push_back(queues_);
  // Check a single net is trained at a time per process, whether single
  // or multi solver. This might also happen if two data layers have same
  // name and same source.
  CHECK(body_->reader_queues_.size() <= solver_count);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      reader_queues_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  try {
    // Synchronize with main thread to make sure we see at least one queue
    {
      boost::mutex::scoped_lock lock(bodies_mutex_);
      CHECK_GE(reader_queues_.size(), 1);
    }
    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so to allow the root solver to start before the other solvers are
    // created, read one item.
    int index = 0;
    if (param_.phase() == TRAIN) {
      read_one(cursor.get(), index++);

      // Wait on remaining solvers
      while (!must_stop()) {
        usleep(100 * 1000);
        boost::mutex::scoped_lock lock(bodies_mutex_);
        if (reader_queues_.size() == solver_count) {
          break;
        }
      }
    }
    // Main loop
    while (!must_stop()) {
      if (index == reader_queues_.size()) {
        index = 0;
      }
      read_one(cursor.get(), index++);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, int index) {
  Datum* datum = reader_queues_[index]->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  reader_queues_[index]->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

}  // namespace caffe
