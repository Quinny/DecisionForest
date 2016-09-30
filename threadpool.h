#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <future>
#include <queue>
#include <thread>

namespace qp {
namespace threading {

template <typename ReturnValue>
class Threadpool {
 public:
  using WorkType = std::function<ReturnValue()>;

  // Starts a thread pool with the number of threads available on the machine.
  Threadpool() : Threadpool(std::thread::hardware_concurrency()) {}

  // Starts a thread pool with the specified number of threads.  Be careful as
  // creating too many threads will have adverse performance benefits.
  Threadpool(int n_threads);

  // Add work to the queue.  Returns a future to the results.
  std::future<ReturnValue> add(const WorkType& work);

  // Shuts down the threadpool.  All tasks currently being executed will finish
  // and all threads will be joined.  All tasks still in the queue will be
  // aborted, and their futures will be invalidated.
  ~Threadpool();

 private:
  std::vector<std::thread> threads_;
  bool shutdown_ = false;
  std::queue<std::pair<WorkType, std::promise<ReturnValue>>> work_queue_;
  std::mutex mu_;
  std::condition_variable work_added_;

  // Worker function which actually carries out the tasks.
  void worker();
};

template <typename T>
Threadpool<T>::Threadpool(int n_threads) {
  for (int i = 0; i < n_threads; ++i) {
    threads_.emplace_back(&Threadpool::worker, this);
  }
}

template <typename T>
std::future<T> Threadpool<T>::add(const WorkType& work) {
  std::lock_guard<std::mutex> lock(mu_);
  work_queue_.emplace(work, std::promise<T>{});
  work_added_.notify_one();
  return work_queue_.back().second.get_future();
}

// TODO - Does not work when T is void.
template <typename T>
void Threadpool<T>::worker() {
  while (true) {
    std::unique_lock<std::mutex> lock(mu_);
    if (shutdown_) {
      return;
    }

    if (work_queue_.empty()) {
      work_added_.wait(lock, [&] { return !work_queue_.empty() || shutdown_; });
    } else {
      auto work_item = std::move(work_queue_.front());
      work_queue_.pop();
      lock.unlock();
      const auto result = work_item.first();
      work_item.second.set_value(result);
    }
  }
}

template <typename T>
Threadpool<T>::~Threadpool() {
  mu_.lock();
  shutdown_ = true;
  work_added_.notify_all();
  mu_.unlock();

  for (auto& thread : threads_) {
    thread.join();
  }

  while (!work_queue_.empty()) {
    work_queue_.front().second.set_exception(std::exception_ptr{});
    work_queue_.pop();
  }
}

}  // namespace threading
}  // namespace qp

#endif /* THREADPOOL_H */
