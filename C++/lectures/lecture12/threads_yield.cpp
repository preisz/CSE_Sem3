// g++ -std=c++17 -Wall -Werror -pedantic -pthread threads_yield.cpp 
// ./a.out & wait
#include <chrono>
#include <thread>
#include <iostream>

int main() {

  auto task = [](int count) {
    // from cppref:
    int yields = 0;
    while(count > yields++)
    {
        // std::this_thread::yield();
    }
  };

  std::thread thread1(task, 1e7);
  std::thread thread2(task, 1e7);
  std::thread thread3(task, 1e7);

  
  thread3.join();
  std::cout << "... thread3 joined back to main thread." << std::endl;
  thread2.join();
  std::cout << "... thread2 joined back to main thread." << std::endl;
  thread1.join();
  std::cout << "... thread1 joined back to main thread." << std::endl;

  return 0;
}