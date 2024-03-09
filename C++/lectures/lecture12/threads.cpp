// g++ -std=c++17 -Wall -Werror -pedantic -pthread threads.cpp 
// ./a.out & wait
#include <chrono>
#include <thread>
#include <iostream>

int main() {

  auto task = [](int duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(duration));    
    std::cout << "task started ... " << std::endl;
    std::cout << "duration=" << duration << std::endl;
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