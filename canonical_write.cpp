#include "canonical.cpp"
#include <fstream>
#include <string>

int main() {
    vi32 shape{8,4,3};
    timept st=time();
    vTensor ret=all_canonicals(shape);
    std::cout<<"time="<<seconds_since(st)<<" sec"<<std::endl;

    std::string name="data/canonicals_";
    for (i32 d=0; d<shape.size(); d++) {
        if (d>0)
            name.append("x");
        name.append(std::to_string(shape.at(d)));
    }
    name.append(".txt");
    std::ofstream out;
    out.open(name);
    for (const Tensor &T:ret) {
        // make parsing as simple as possible
        for (u8 v:T.flattened())
            out<<std::to_string(v);
        out<<std::endl;
    }
    out.close();
}