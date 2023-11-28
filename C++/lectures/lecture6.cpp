#include <utility>

// general includes
#include <iostream>   // std::cout|endl 
#include <vector>   // std::vector
#include <list>   // std::list
#include <string>
// using namespace std;
struct NoWidget {};
struct WidgetM {
    int m;
};

template <typename FIRST, typename SECOND>
struct PairTT {
  FIRST first;
  SECOND second;
  PairTT(const FIRST &first, const SECOND &second)
      : first(first), second(second) {}
  auto operator+(const PairTT &other) { 
    return PairTT{first + other.first, second + other.second};
  }
};

// usage 
PairTT p1(1, 2.0);  // (1) instantiates Pair<int, double> 
//PairTT p2(2.0, 1);  // (2) instantiates Pair<double, int> 
//auto sum = p1 + p2; // (3) still works ? AUTO> void func(AUTO&& arg){ otherFunc(;) }

int main(void){
PairTT p1(1, 2.0);  // (1) instantiates Pair<int, double> 
PairTT p2(2.0, 1);  // (2) instantiates Pair<double, int> 
//auto sum = p1 + p2; // (3) still works ? NO, no conversion



    return EXIT_SUCCESS;
}