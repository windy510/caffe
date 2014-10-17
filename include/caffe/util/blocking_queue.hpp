#ifndef CAFFE_UTIL_BLOCKING_QUEUE_H_
#define CAFFE_UTIL_BLOCKING_QUEUE_H_

#include <queue>
#include "boost/thread.hpp"

namespace caffe {

template<typename T>
class blocking_queue {
public:
  void push(const T& t) {
    boost::mutex::scoped_lock lock(mutex_);
    queue_.push(t);
    lock.unlock();
    condition_.notify_one();
  }

  bool empty() const {
    boost::mutex::scoped_lock lock(mutex_);
    return queue_.empty();
  }

  bool try_pop(T& t) {
    boost::mutex::scoped_lock lock(mutex_);

    if (queue_.empty())
      return false;

    t = queue_.front();
    queue_.pop();
    return true;
  }

  T pop(bool show = false) {
    boost::mutex::scoped_lock lock(mutex_);

    while (queue_.empty())
      condition_.wait(lock);

    T t = queue_.front();
    queue_.pop();

//    if(show)
//      LOG(INFO)<<"pop, size - " <<queue_.size();

    return t;
  }

private:
  std::queue<T> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

}  // namespace caffe

#endif
