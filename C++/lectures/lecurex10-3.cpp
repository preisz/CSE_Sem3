#include <iostream>

struct Widget{
    int m;
};

template <typename T>
struct UniquePtr {
    T* ptr;

    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;
    UniquePtr(T* ptr = nullptr) : ptr(ptr) {}

    // Move constructor
    UniquePtr(UniquePtr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }

    // Move assignment
    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr; other.ptr = nullptr;
        }
        return *this;
    }

    ~UniquePtr() {delete ptr; }

    T& operator*() const {  assert(ptr != nullptr); return *ptr; }
    T* operator->() const {  assert(ptr != nullptr);  return ptr; }

};

template <typename T>
void pass_ownership(UniquePtr<T>&& other) {
    // Do something with the ownership transfer, if needed
}

int main(){
    UniquePtr<Widget> up(new Widget[2]);
    // UniquePtr<Widget> up2 = up; // This line won't compile due to the deleted copy constructor
    UniquePtr<Widget> up2 = std::move(up); // Use move constructor

    //pass_ownership(std::move(up));
    int check = (up.ptr == nullptr); // Expect that after the line above
    std::cout << "++++++++++++++++++++++++++++      " <<check;

    return EXIT_SUCCESS;
}