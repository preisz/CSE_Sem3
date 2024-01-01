#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct Base {
  virtual std::string interface_function() const { return "Base"; };
  virtual ~Base() {
    delete m;
    std::cout << "base::dtor" << std::endl;
  };
  int *m;
  Base() : m(new int){};
};

struct EngineA : public Base {
  std::string interface_function() const override { return "A"; }
  ~EngineA() {
    delete m;
    std::cout << "EngineA::dtor" << std::endl;
  };
  int *m;
  EngineA() : m(new int){}; // only allow derived classed to construct
};

struct EngineB : public Base {
  std::string interface_function() const override { return "B"; }
  ~EngineB() {
    delete m;
    std::cout << "EngineB::dtor" << std::endl;
  };
  int *m;
  EngineB() : m(new int){}; // only allow derived classed to construct
};

int main() {

  //   std::vector<Base *> vec;
  std::vector<std::unique_ptr<Base>> vec;

  vec.push_back(std::make_unique<EngineA>());

  for (const auto &item : vec)
    std::cout << item->interface_function() << std::endl;

  return 0;
}
