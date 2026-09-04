#include "search_utils.cpp"

void test_tensor() {
    {
        // == operator
        assert(Tensor({2},{0,1})==Tensor({2},{0,1}), "equal");
        assert(!(Tensor({2},{0,1})==Tensor({1,2},{0,1})), "not equal shape");
        assert(!(Tensor({2},{0,1})==Tensor({2},{0,0})), "not equal elements");
    }
    {
        // general instance methods
        Tensor T({1,2,3},{0,1,0,0,1,1});
        assert(T==T.copy(), "copy");
        assert(T.ndim()==3, "ndim");
        assert(T.len(0)==1 && T.len(1)==2 && T.len(2)==3, "len");
        assert(T.size()==6, "size");
        assert(T.at(0)==0 && T.at(1)==1 && T.at(2)==0 && T.at(3)==0 && T.at(4)==1 && T.at(5)==1, "at");
        assert(T.flattened()==vu8{0,1,0,0,1,1}, "flattened");
    }
    {
        // mat_at
        Tensor T({2,3},{0,1,0,0,1,1});
        assert(T.mat_at(0,0)==0 && T.mat_at(0,1)==1 && T.mat_at(0,2)==0 && T.mat_at(1,0)==0 && T.mat_at(1,1)==1 && T.mat_at(1,2)==1, "mat_at");
    }
}
void test_tensor_enum() {
    // check exact order so that we don't accidentally change search order of DFS
    {
        // tensor
        assert(all_tensors({})==vTensor{Tensor({},{0}),Tensor({},{1})}, "all_tensors 0-dim");
        assert(all_tensors({0})==vTensor{Tensor({0},{})}, "all_tensors size 0");
        assert(all_tensors({1,2})==vTensor{Tensor({1,2},{0,0}),Tensor({1,2},{0,1}),Tensor({1,2},{1,0}),Tensor({1,2},{1,1})}, "all_tensors general");
    }
    {
        // normalized vectors
        assert(all_normalized_vecs(1)==vTensor{Tensor({1},{1})}, "all_normalized_vecs n=1");
        assert(all_normalized_vecs(2)==vTensor{Tensor({2},{0,1}),Tensor({2},{1,0}),Tensor({2},{1,1})}, "all_normalized_vecs n=2");
    }
}
void test_axis0_op() {
    {
        assert(
            axis0_op(
                Tensor(
                    {2,2},
                    {
                        1,1,
                        0,1,
                    }
                ),
                Tensor({2,3},{0,1,1, 1,1,0})
            )
            ==Tensor({2,3},{1,0,1, 1,1,0}),
            "axis0_op, general"
        );
    }
    {
        assert(axis0_op(Tensor({2,0},{}),Tensor({0,3},{}))==Tensor({2,3},vu8(6,0)), "axis0_op, inner length = 0");
    }
    {
        assert(
            axis0_op(
                Tensor(
                    {2,3},
                    {
                        1,1,1,
                        1,1,0,
                    }
                ),
                Tensor({3},{0,1,1})
            )
            ==Tensor({2},{0,1}),
            "axis0_op, target tensor 1d"
        );
    }
}
bool has_full_axis0_rank(const Tensor &T) {
    for (const Tensor &v:all_normalized_vecs(T.len(0))) {
        Tensor contraction=axis0_op(Tensor({1,T.len(0)},v.flattened()),T);
        bool all_zero=true;
        for (i32 i=0; i<contraction.size(); i++)
            if (contraction.at(i)!=0) {
                all_zero=false;
                break;
            }
        if (all_zero)
            return false;
    }
    return true;
}
void test_mat_row_reduce() {
    {
        // rank 0
        Tensor T({2,3},vu8(6,0));
        RowReduce ret=mat_row_reduce(T);
        assert(ret.rank==0, "mat_row_reduce, rank 0 check rank");
        assert(
            axis0_op(ret.reducer,T)
            ==Tensor({2,3},vu8(6,0)),
            "mat_row_reduce, rank 0 check reducer"
        );
        assert(has_full_axis0_rank(ret.reducer), "mat_row_reduce, rank 0 check reducer rank");
    }
    {
        // rank 1
        Tensor T({3,4},{0,0,0,0, 0,1,1,1, 0,1,1,1});
        RowReduce ret=mat_row_reduce(T);
        assert(ret.rank==1, "mat_row_reduce, rank 1 check rank");
        assert(
            axis0_op(ret.reducer,T)
            ==Tensor({3,4},{0,1,1,1, 0,0,0,0, 0,0,0,0}),
            "mat_row_reduce, rank 1 check reducer"
        );
        assert(has_full_axis0_rank(ret.reducer), "mat_row_reduce, rank 1 check reducer rank");
    }
    {
        // general
        Tensor T({4,4},{1,0,1,1, 1,1,0,0, 1,1,0,1, 0,1,1,0});
        RowReduce ret=mat_row_reduce(T);
        assert(ret.rank==3, "mat_row_reduce, general check rank");
        assert(
            axis0_op(ret.reducer,T)
            ==Tensor({4,4},{1,0,1,0, 0,1,1,0, 0,0,0,1, 0,0,0,0}),
            "mat_row_reduce, general check reducer"
        );
        assert(has_full_axis0_rank(ret.reducer), "mat_row_reduce, rank 0 check reducer rank");
    }
}
void test_concise() {
    typedef struct {
        Tensor T;
        vi32 axis_ranks;
    } Test;
    std::vector<Test> tests{
        {
            .T=Tensor({0,0},{}),
            .axis_ranks={0,0},
        },
        {
            .T=Tensor({2,3,4},vu8(2*3*4,0)),
            .axis_ranks={0,0,0},
        },
        {
            .T=Tensor({1},{1}),
            .axis_ranks={1},
        },
        {
            .T=Tensor({2,2},{1,0, 0,1}),
            .axis_ranks={2,2},
        },
        {
            .T=Tensor({2,3},{1,0,1, 0,1,1}),
            .axis_ranks={2,2},
        },
        {
            .T=Tensor({2,2,2},{0,1,1,0, 1,0,0,0}),
            .axis_ranks={2,2,2},
        },
    };
    for (const Test &test:tests) {
        const Tensor &T=test.T;
        ConciseTensor ret=concise(T);
        assert(ret.concise.get_shape()==test.axis_ranks, "test_concise: wrong shape");
        Tensor expanded=ret.concise;
        for (i32 ax=0; ax<T.ndim(); ax++)
            expanded=axis_op(ret.expanders.at(ax),expanded,ax);
        assert(expanded==T, "test_concise: wrong tensors");
    }
}
void test_nCr() {
    assert(nCr(0,0)==1, "nCr n=0");
    assert(nCr(1,0)==1, "nCr k=0");
    assert(nCr(2,2)==1, "nCr k=n");
    assert(nCr(2,3)==0, "nCr k>n");
    assert(nCr(5,2)==10, "nCr nontrivial");
    assert(nCr(100,15)==253338471349988640LL, "nCr large");
    assert(nCr(100,85)==253338471349988640LL, "nCr large k>n/2");
}
void test_packed() {
    Tensor T({1,2,3},{0,1,0,0,1,1});
    assert(packed(T)==0b110010, "packed");
}
void test_outer_prod() {
    {
        // 2 vectors
        assert(
            outer_prod({Tensor({2},{1,0}),Tensor({3},{0,1,1})})
            ==Tensor(
                    {2,3},
                    {
                        0,1,1,
                        0,0,0
                    }
            ),
            "outer_prod 2d"
        );
    }
    {
        // 3 vectors
        assert(
            outer_prod({Tensor({2},{1,0}),Tensor({2},{0,1}),Tensor({3},{0,1,1})})
            ==Tensor(
                {2,2,3},
                {
                    0,0,0,
                    0,1,1,

                    0,0,0,
                    0,0,0,
                }
            ),
            "outer_prod 3d"
        );
    }
}
void test_is_valid_cpd() {
    assert(is_valid_cpd(Tensor({2,3},vu8(6,0)),{}), "CPD rank 0");
    assert(is_valid_cpd(Tensor({2,3},vu8(6,1)),vvTensor{{Tensor({2},vu8(2,1)),Tensor({3},vu8(3,1))}}), "CPD rank 1");
    assert(!is_valid_cpd(Tensor({2,3},{1,1,1, 1,1,0}),vvTensor{{Tensor({2},vu8(2,1)),Tensor({3},vu8(3,1))}}), "CPD not rank 1");
}

void test_cube_set() {
    {
        CubeSubset cube=CubeSubset::all_ptwise_le(3,5,{});
        assert(!cube.contains({0,1,2,2,1}), "cube set empty");
    }
    {
        CubeSubset cube=CubeSubset::all_ptwise_le(3,5,{
            {0,1,2,2,1},
        });
        assert(cube.contains({0,1,2,2,1}), "cube set singleton: match");
        assert(cube.contains({0,0,0,0,0}), "cube set singleton: ptwise >=");
        assert(!cube.contains({1,0,0,0,0}), "cube set singleton: ptwise < at start");
        assert(!cube.contains({0,0,0,0,2}), "cube set singleton: ptwise < at end");
    }
    {
        CubeSubset cube=CubeSubset::all_ptwise_le(3,5,{
            {0,1,2,2,1},
            {0,1,1,1,2},
        });
        assert(cube.contains({0,1,2,2,1}), "cube set multi: match old");
        assert(cube.contains({0,1,1,1,2}), "cube set multi: match new");
        assert(cube.contains({0,0,0,0,0}), "cube set multi: ptwise >=");
        assert(!cube.contains({0,1,2,2,2}), "cube set multi: ptwise < at end");
    }
}

void test_search_init_speed() {
    std::chrono::steady_clock::time_point st=time();
    Search(4,4,4);
    std::cout<<"init 4x4x4: "<<seconds_since(st)<<" sec"<<std::endl;
}

void test_search() {
    std::vector<CPDTest> tests{
        {
            .T=Tensor({1,1,1},{1}),
            .true_rank=1,
        },
        {
            // superdiagonal
            .T=Tensor({2,2,2},{1,0,0,0, 0,0,0,1}),
            .true_rank=2,
        },
        {
            // concise tensor with all side lengths different
            .T=Tensor({4,3,2},{
                1,0,1,
                0,1,1,

                1,0,1,
                0,1,0,

                1,0,1,
                0,0,0,

                1,0,0,
                0,0,0,
            }),
            .true_rank=4,
        },
        {
            // isomorphic to W-state over characteristic 2
            .T=Tensor({2,2,2},{1,0,0,1, 0,1,1,0}),
            .true_rank=3,
        },
        {
            .T=Tensor(
                {3,3,3},
                {
                    0,0,1,
                    0,1,0,
                    1,0,0,

                    0,1,0,
                    1,0,0,
                    0,0,0,

                    1,0,0,
                    0,0,0,
                    0,0,0,
                }
            ),
            .true_rank=5,
        },
        {
            // Strassen
            .T=Tensor(
                {4,4,4},
                {
                    1,0,0,0,
                    0,0,1,0,
                    0,0,0,0,
                    0,0,0,0,

                    0,0,0,0,
                    0,0,0,0,
                    1,0,0,0,
                    0,0,1,0,

                    0,1,0,0,
                    0,0,0,1,
                    0,0,0,0,
                    0,0,0,0,

                    0,0,0,0,
                    0,0,0,0,
                    0,1,0,0,
                    0,0,0,1,
                }
            ),
            .true_rank=7,
        },
    };
    for (const CPDTest &test:tests) {
        test_search_tensor(test.T,test.true_rank,test.true_rank+2);
    }
}

int main() {
    test_tensor();
    test_tensor_enum();
    test_axis0_op();
    test_mat_row_reduce();
    test_concise();
    test_nCr();
    test_packed();
    test_outer_prod();
    test_is_valid_cpd();
    test_cube_set();
    test_search_init_speed();
    test_search();
}