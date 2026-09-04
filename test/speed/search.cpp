#include "../search_utils.cpp"

int main() {
    std::vector<CPDTest> tests{
        {
            .T=Tensor(
                {4,4,4},
                {
                    1,0,0,0,
                    0,0,0,0,
                    0,0,0,0,
                    0,0,0,0,

                    0,1,0,0,
                    1,0,0,0,
                    0,0,0,0,
                    0,0,0,0,

                    0,0,1,0,
                    0,1,0,0,
                    1,0,0,0,
                    0,0,0,0,

                    0,0,0,1,
                    0,0,1,0,
                    0,1,0,0,
                    1,0,0,0,
                }
            ),
            .true_rank=8,
        },
        {
            .T=Tensor(
                {4,4,4},
                {
                    0,0,0,0,
                    0,0,0,0,
                    0,0,0,1,
                    0,0,1,0,

                    0,0,0,0,
                    0,0,0,0,
                    0,0,1,0,
                    0,0,1,1,

                    0,0,0,1,
                    0,0,1,0,
                    0,1,0,0,
                    1,0,0,0,

                    0,0,1,0,
                    0,0,1,1,
                    1,0,0,0,
                    1,1,0,0,
                }
            ),
            .true_rank=9,
        },
    };
    for (const CPDTest &test:tests) {
        test_search_tensor(test.T,test.true_rank,test.true_rank);
    }
}