//object in c++ :: class behinf´d it

struct  Widget
{
    double a;
    int i;
    //what to do when copy?...
};

struct Vector //very fundamental, not practical impl of vect
{
    double* data; //points to dynamic memory
    int n; //current size
}; //provide own logic



void func(std::list <double>& lst){ //no built in ref counting in C++

}

int main(){
    Vector vec1 =...;

    return EXIT_SUCCESS;
}