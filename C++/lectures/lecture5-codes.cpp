
struct  Vector{

double *_data = nullptr;
size_t _size =0:

Vector(size_t n, double init) : _data(new double[n]), size_t n { 
    //init 
}
~Vector() { delete(data) }
Vector(const Vector &other);
Vector(Vector&& other );
Vector& operator=(const Vector& other) {}
Vector& operator=(Vector&& other) {}

};

//Wrapper func: good, b.c you can be very generic
void wrap(T arg){
    func(arg);
}


 

int main(){

    Vector vec;
    Vector vec(10, 0.0);

    // copy ctor/asign: LHS does not exists
    Vector v2 = vec; //func(Vector v2)
    Vector v3 = std::move(v2);  //func(Vector&& v2 )

    //func333(const Vector& v2): Binding to the reference

    // move ctor/assig: LHS already exists!
    v2 = vec;
    v3 = std::move(vec);
    Vector v2 = Vector(10, 1.0); //probanly copied- ostum constructor

    //#################
    func(expr);
    wrap(expr); //func should not realize
    //ehther it is called inside wrap or not

    //temporary: construct on spot
    func(Widget());

    //Not temporary
    Widget expr;
    func(expr);

    return EXIT_SUCCESS;


}