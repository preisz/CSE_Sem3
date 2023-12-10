#include <iostream>
#include <vector>
#include <algorithm>

struct Widget{

};

using FuncPtr = bool (*) (const Widget&);

struct State
{
    int m;
};

bool func(const Widget& w){return m < w.m;}
bool func2(State *self, const Widget& w){return m < w.m;}


int main(){
    Widget w;

    struct Compare //inline class, create on the craft
    {
        int m;
        void func(){};

        bool operator()(const Widget& w){
            return m < w.m ? true : false; //this means if true return first one if not return second one
        //equivalent to
        // if (m<w.m){return true}; else{return false;}
        } //can create on the spot things
    
    };

    Compare o; 
    o.m = 7; 
    
    std::cout << o.m << std::endl;

    //Inline functions should be not allowe
    o.func(); //works , even though it is an inline class
    //apparently inline class works this way

    std::vector<Widget> vec = { {1}, {2}, {3} };

    auto res = std::count_if(std::begin(vec),
        std::end(vec), Check{1} ); 

    FuncPtr checker = func;
    auto res2 = std::count_if(std::begin(vec),
        std::end(vec), checker ); 

    
}