#include "../../canonical.cpp"

int main() {
    vvi32 shapes{
        {4,4,3},
    };
    for (const vi32 &shape:shapes) {
        std::cout<<"================================"<<std::endl;
        std::cout<<"speedtesting shape="<<shape<<std::endl;
        timept st=time();
        vTensor ret=all_canonicals(shape);
        std::cout<<"time="<<seconds_since(st)<<" sec"<<std::endl;
    }
}