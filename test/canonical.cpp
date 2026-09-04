#include "../canonical.cpp"

void test_Tensor() {
    {
        Tensor mat({4,5},{
            1,1,1,1,0,
            1,1,1,0,0,
            1,0,0,1,1,
            0,1,1,1,1,
        });
        Tensor::RowReduceRet ret=Tensor::rref(mat);
        assert(
            ret.rref==Tensor(
                {4,5},
                {
                    1,0,0,0,1,
                    0,1,1,0,1,
                    0,0,0,1,0,
                    0,0,0,0,0,
                }
            ),
            "test_Tensor rref: wrong rref"
        );
        assert(ret.rank==3, "test_Tensor rref: wrong rank");
        assert(
            mat==Tensor({4,5},{
                1,1,1,1,0,
                1,1,1,0,0,
                1,0,0,1,1,
                0,1,1,1,1,
            }),
            "test_Tensor rref: mutated input"
        );
        assert(Tensor::axis_op(ret.reducer,mat,0)==ret.rref, "test_Tensor reducer: inconsistent with rref");
    }
    {
        assert(
            Tensor::all_tensors_lex({1,2})
            ==vTensor({
                Tensor({1,2},{0,0}),
                Tensor({1,2},{0,1}),
                Tensor({1,2},{1,0}),
                Tensor({1,2},{1,1}),
            }),
            "test_Tensor all_tensors_lex"
        );
    }
    {
        for (i32 n=2; n<4; n++) {
            vu8 I_data;
            for (i32 i=0; i<n; i++)
                for (i32 j=0; j<n; j++)
                    I_data.push_back(i==j?1:0);
            Tensor I({n,n},I_data);
            for (const Tensor &M:Tensor::all_tensors_lex({n,n})) {
                bool is_invle=(mat_rank(M)==n);
                std::optional<Tensor> ret=mat_inv(M);
                assert(ret.has_value()==is_invle, "test_Tensor mat_inv: incorrect existence of solution");
                if (is_invle)
                    assert(Tensor::axis_op(ret.value(),M,0)==I, "test_Tensor mat_inv: incorrect value");
            }
        }
        {
            Tensor M({3,4},{
                1,0,0,1,
                0,1,0,1,
                0,0,1,1,
            });
            // rref([M | I]) has identity matrix as left prefix
            std::optional<Tensor> ret=mat_inv(M);
            assert(!ret.has_value(), "test_Tensor mat_inv: incorrectly accepted non-square matrix");
        }
    }
}
void test_all_canonicals() {
    assert(
        all_canonicals({1,2,3})
        ==vTensor{
            Tensor::zeros({1,2,3}),
            Tensor({1,2,3},{0,0,0, 0,0,1}),
            Tensor({1,2,3},{0,0,1, 0,1,0}),
        },
        "all_canonicals 1x2x3"
    );
    assert(
        all_canonicals({2,2,2})
        ==vTensor{
            Tensor({2,2,2},vu8(2*2*2,0)),
            Tensor({2,2,2},{0,0,0,0, 0,0,0,1}),
            // axis-ranks (1,2,2) + permutations
            Tensor({2,2,2},{0,0,0,0, 0,1,1,0}),
            Tensor({2,2,2},{0,0,0,1, 0,0,1,0}),
            Tensor({2,2,2},{0,0,0,1, 0,1,0,0}),
            // W-state
            Tensor({2,2,2},{0,0,0,1, 0,1,1,0}),
            // superdiagonal
            Tensor({2,2,2},{0,0,0,1, 1,0,0,0}),
            // all nonzero axis-0 contractions have rank 2
            Tensor({2,2,2},{0,1,1,0, 1,0,1,1}),
        },
        "all_canonicals 2x2x2"
    );
    assert(all_canonicals({3,2,2}).size()==10, "all_canonicals 3x2x2 count");
    assert(all_canonicals({3,3,2}).size()==21, "all_canonicals 3x3x2 count");
    assert(all_canonicals({2,3,4}).size()==28, "all_canonicals 2x3x4 count");
    assert(all_canonicals({3,3,3}).size()==116, "all_canonicals 3x3x3 count");
    assert(all_canonicals({4,3,3}).size()==355, "all_canonicals 4x3x3 count");
}
int main() {
    std::cout<<"sizeof(Tensor)="<<sizeof(Tensor({},vu8{}))<<std::endl;
    test_Tensor();
    test_all_canonicals();
}