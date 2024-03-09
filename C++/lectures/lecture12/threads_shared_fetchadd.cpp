// g++ -std=c++17 -Wall -Werror -pedantic -pthread threads_shared_fetchadd.cpp
// ./a.out
#include <chrono>
#include <iostream>
#include <atomic>
#include <string>
#include <thread>

int main() {

  std::atomic<int> shared = 1; // shared int

  auto task = [&shared](int duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(duration));
    std::cout << "task started ... " << std::endl;
    std::cout << "duration=" << duration << std::endl;
    { // inc/dec shared int
      // shared.fetch_add(1,std::memory_order_seq_cst);
      shared.fetch_add(1,std::memory_order_relaxed);
    }
    std::cout << "reading_shared_int: " << shared << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(duration * 1000));
    std::cout << "... task completed." << std::endl;
  };

  std::thread thread1(task, 1);
  std::thread thread2(task, 2);
  std::thread thread3(task, 3);

  thread3.join();
  std::cout << "... thread3 joined back to main thread." << std::endl;
  thread2.join();
  std::cout << "... thread2 joined back to main thread." << std::endl;
  thread1.join();
  std::cout << "... thread1 joined back to main thread." << std::endl;

  return 0;
}