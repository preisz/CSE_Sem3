// g++ -std=c++17 -Wall -Werror -pedantic processes.cpp 
// ./a.out 3 & ./a.out 2 & ./a.out 1 & wait

#include <chrono>
#include <iostream>
#include <cstdlib>
#include <thread>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "usage: ./a.out DURATION_IN_SECONDS " << std::endl;
    return 0;
  }

  int duration = std::atoi(argv[1]);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  std::cout << "process started ... " << std::endl;
  std::cout << "duration=" << duration << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(duration * 1000));
  std::cout << "... process completed." << std::endl;

  return 0;
}