#include <memory>

template <class T> class shared_ptr {
  struct ControlBlock {
    int count = 1;
    // some more members in stdlib
  };
  ControlBlock *cb; // (1) member A -> for refcounting
  T *ptr;           // (2) member B -> same as for unqiue_ptr

  void increment() {
    if (cb)
      ++cb->count;
  }
  void decrement() {
    if (cb && --cb->count == 0) {
      delete ptr; // wrapped resource
      delete cb;  // overhead
    }
  }

public:
  // usage: auto sp = shared_ptr<Widget>(new Widget{});
  shared_ptr(T *ptr) : cb(new ControlBlock()), ptr(ptr) {}
  shared_ptr(const shared_ptr &other) : cb(other.cb), ptr(other.ptr) {
    // (1) ???
    // 1) setup sharing of refcount and resource
    // 2) increment
  }
  shared_ptr(shared_ptr &&other) : cb(other.cb), ptr(other.ptr) {
    // (2) ???
    // 1) no inc/dec
    // 2) but: setup sharing of refcount and resource + set 'other' to
    // "NULL"
  }
  shared_ptr &operator=(const shared_ptr &other) {
    // (3) ???
    // 1) release managing the current resource (dec) + check if delete
    // 2) take over members from 'other' + inc
    return *this;
  }
  shared_ptr &operator=(shared_ptr &&other) {
    // (4) ???
    // 1) release managing the current resource (dec) + check if delete
    // 2) take over members from 'other' + set other to "NULL"
  }
  ~shared_ptr() {
    // (5) ???
    // 1) check if we are already the last owner (unique owner): if yes
    // free resource
    // 2) if not: decrement
  }
  T *operator->() const { return ptr; }
  T &operator*() const { return *ptr; }
};

template <typename T, typename... ARGS>
shared_ptr<T> make_shared(ARGS &&... args) {
  // do sth. here: -> use a single allocation for cb and ptr
  return shared_ptr<T>(new T(std::forward<ARGS>(args)...));
}
