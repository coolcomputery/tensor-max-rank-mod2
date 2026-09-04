#include "../search.cpp"

bool is_valid_cpd(const Tensor &T, const vvTensor &cpd) {
    vu8 data(T.size(),0);
    for (const vTensor &tup:cpd) {
        Tensor prod=outer_prod(tup);
        for (i32 i=0; i<T.size(); i++)
            data[i]=(data.at(i)+prod.at(i))%MOD;
    }
    return Tensor(T.get_shape(),data)==T;
}
void test_search_tensor(Tensor T, i32 true_rank, i32 max_test_rank) {
    for (i32 rep=0; rep<64; rep++)
        std::cout<<"=";
    std::cout<<std::endl;
    std::cout<<"T="<<T<<std::endl;

    timept init_st=time();
    Search s(T.len(0),T.len(1),T.len(2));
    std::cout<<"init sec="<<seconds_since(init_st)<<std::endl;

    for (i32 R=0; R<=max_test_rank; R++) {
        timept st=time();
        Search::Result ret=s.search(T,R);
        std::cout<<"\tR="<<R
            <<" sec="<<seconds_since(st)
            <<" nvisited="<<ret.nvisited
            <<" npruned="<<ret.npruned
            <<std::endl;
        std::optional<vvTensor> cpd=ret.cpd;
        std::cout<<"\tcpd="<<cpd<<std::endl;
        if (R<true_rank)
            assert(!cpd.has_value(), "CPD unexpectedly found");
        else {
            assert(cpd.has_value(), "no CPD found");
            assert(is_valid_cpd(T,cpd.value()),"incorrect CPD");
        }
    }
}
typedef struct {
    Tensor T;
    i32 true_rank;
} CPDTest;