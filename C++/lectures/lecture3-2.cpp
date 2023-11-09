// compile & inspect: g++ -c func2.cpp -o func2.o && nm func2.o

struct Widget {
    void member();
};

void Widget::member(){};
void non_member(Widget* self){};

void func1(){}
void func2(int a){}
void func3(double b){}
void func4(int a,double b){}