#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct Widget {
  int m;
  virtual int func(){return m +10;};

  virtual ~Widget() = default; //opt in = should serve as base class in polymorphic hierarchy
};

struct WidgetInterface{
    virtual int func() = 0;
    virtual int draw() = 0;
    virtual int save() = 0;
    virtual ~WidgetInterface() = default;
};

struct Other  { //with has -a
private:
  Widget w;

public:
    double d;
    Other(int a) : d(1.0), w(a) {}; //ctor
};

struct Other2 : public Widget { //also hides Widget if private
    double d;

public:
    //using Widget::Widget; //expose public
    //int func(){Widget::func(); //use public interface of Widget}

    Other2(int a) : Widget(a), d(1.0) {}
    int func() override {return m +20;} //explicitely tell that upper in hierarchy, overloaded

    ~Other2(){/*important stuff*/}
};

struct OtherInterface : public WidgetInterface{
    int func() override {return 2;};
    int draw() override {return 21;};
    int save() override {return 42;};
};
struct Button : public WidgetInterface{
    int func() override {return 1;};
    int draw() override {return 1;};
    int save() override {return 2;};
};

int main(){

    Other o(1);
    o.w.func(); //inaccesible

    Other2 o2(1);

    Widget wb = o2; //copy ctor widget part
    Widget *b = &o2;

    b -> func(); // +20

    Other2 *o2 = static_cast<Other2 *>(b); //static cast: no overhead at runtime, happens at compil time
    Other2 *o21 = dynamic_cast<Other2 *>(b); //runtime

    Widget *b2 = new Other2{};
    delete b; //call dtor of base class, only of base, if note defined as virtual!!

    std::vector<WidgetInterface *> drawables;
    drawables.append(new Button{});
    drawables.append(new OtherInterface{});

    for (auto item : drawables)
        item ->draw();
    
    auto copy = drawables; //copies what it holds also the pointers!
    //the copied pointers point to same things

    for (auto item : drawables)
        delete item;
    

    return EXIT_SUCCESS;

}