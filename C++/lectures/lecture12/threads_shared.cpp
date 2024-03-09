// g++ -std=c++17 -Wall -Werror -pedantic -pthread threads_shared.cpp
// ./a.out
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main() {

  std::string shared = "shared buffer";

  auto task = [&shared](int duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(duration));
    std::cout << "task started ... " << std::endl;
    std::cout << "duration=" << duration << std::endl;
    { // accessing shared buffer
      shared = shared + "|| unsynchronized ||";
      std::cout << "reading_shared_buffer: " << shared << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(duration * 10));
    std::cout << "... task completed." << std::endl;
  };

  std::thread thread1(task, 1);
  std::thread thread2(task, 1);
  std::thread thread3(task, 1);

  thread3.join();
  std::cout << "... thread3 joined back to main thread." << std::endl;
  thread2.join();
  std::cout << "... thread2 joined back to main thread." << std::endl;
  thread1.join();
  std::cout << "... thread1 joined back to main thread." << std::endl;

  return 0;
}