#include "search.cpp"
#include <fstream>

bool is_valid_cpd(const Tensor &T, const vvTensor &cpd) {
    vu8 data(T.size(),0);
    for (const vTensor &tup:cpd) {
        Tensor prod=outer_prod(tup);
        for (i32 i=0; i<T.size(); i++)
            data[i]=(data.at(i)+prod.at(i))%MOD;
    }
    return Tensor(T.get_shape(),data)==T;
}
vTensor read_tensors(const vi32 &shape, const std::string &file_name) {
    i32 size=prod(shape);
    vTensor Ts;
    std::ifstream file(file_name);
    assert(file.good(), "read_tensors: file not good (does not exist?)");
    for (std::string line; std::getline(file,line);) {
        assert(line.size()==size, "read_tensors: wrong number of elems");
        vu8 data;
        for (i32 i=0; i<size; i++)
            data.push_back(line.at(i)-'0');
        Ts.push_back(Tensor(shape,data));
    }
    file.close();
    return Ts;
}
int main() {
    vi32 shape{8,4,3};
    vTensor Ts=read_tensors(shape,"data/canonicals_8x4x3.txt");
    std::cout<<"shape="<<shape<<" n_tensors="<<Ts.size()<<std::endl;

    std::map<vi32,Search> searchers;
    i32 max_rank=0;
    timept st=time();
    double time_mark=10;
    u64 cnt=0;
    typedef struct {
        Tensor T;
        i32 rank;
    } RankResult;
    typedef struct {
        Tensor T;
        i32 R;
        std::vector<u64> nvisited;
        std::vector<u64> npruned;
    } SearchStats;  // same as Search::Result but no .cpd
    std::vector<RankResult> ranks;
    std::vector<SearchStats> search_rets;
    for (const Tensor &T:Ts) {
        ConciseTensor compressed=concise(T);
        // use original tensor if already concise
        const Tensor &Tc=compressed.concise.get_shape()==T.get_shape()?T:compressed.concise;
        const vi32 &cshape=Tc.get_shape();
        if (prod(cshape)==0) {
            ranks.push_back({
                .T=T,
                .rank=0,
            });
            search_rets.push_back({
                .T=T,
                .R=0,
                .nvisited={},
                .npruned={},
            });
        }
        else {
            if (searchers.find(cshape)==searchers.end())
                searchers.insert({cshape,Search(cshape.at(0),cshape.at(1),cshape.at(2))});
            i32 R=0;
            for (i32 n:cshape)
                R=std::max(R,n);
            std::optional<vvTensor> cpd;
            while (true) {
                Search::Result ret=searchers.at(cshape).search(Tc,R);
                search_rets.push_back({
                    .T=T,
                    .R=R,
                    .nvisited=ret.nvisited,
                    .npruned=ret.npruned,
                });
                cpd=ret.cpd;
                if (cpd.has_value())
                    break;
                R++;
            }
            if (!is_valid_cpd(Tc,cpd.value())) {
                std::cout<<"INCORRECT CPD: "<<"T="<<Tc<<" "<<"R="<<R<<"\n\t"<<cpd<<std::endl;
                throw std::runtime_error("incorrect CPD");
            }
            ranks.push_back({
                .T=T,
                .rank=R,
            });
            if (R>max_rank) {
                std::cout<<"R="<<R<<" T="<<T<<" "<<"\n\t"<<cpd<<std::endl;
                max_rank=R;
            }
        }

        cnt++;
        double elapsed=seconds_since(st);
        if (elapsed>time_mark) {
            std::cout<<"cnt="<<cnt<<" time="<<elapsed<<std::endl;
            while (elapsed>time_mark)
                time_mark*=2;
        }
    }
    std::cout<<"cnt="<<cnt<<" time="<<seconds_since(st)<<std::endl;
    std::cout<<"max_rank="<<max_rank<<std::endl;

    std::string shape_str="";
    for (i32 d=0; d<shape.size(); d++) {
        if (d>0)
            shape_str.append("x");
        shape_str.append(std::to_string(shape.at(d)));
    }

    // print rank of each tensor
    {
        std::string name="data/ranks_";
        name.append(shape_str);
        name.append(".tsv");
        std::ofstream out;
        out.open(name);
        out<<"T\trank"<<std::endl;
        for (const RankResult &ret:ranks) {
            for (u8 v:ret.T.flattened())
                out<<std::to_string(v);
            out<<"\t"<<ret.rank
                <<std::endl;
        }
        out.close();
    }

    // print search stats for each (tensor, rank) pair
    {
        std::string name="data/cpd_search_stats_";
        name.append(shape_str);
        name.append(".tsv");
        std::ofstream out;
        out.open(name);
        out<<"T\tR\tnvisited\tnpruned"<<std::endl;
        for (const SearchStats &ret:search_rets) {
            for (u8 v:ret.T.flattened())
                out<<std::to_string(v);
            out<<"\t"<<ret.R
                <<"\t"<<ret.nvisited
                <<"\t"<<ret.npruned
                <<std::endl;
        }
        out.close();
    }
}