#include <iostream>
#include "../include/linear-gaussian-system.h"

#define ASSERT_EQ(x, y)                                                \
    {                                                                  \
        if ((x) != (y))                                                \
        {                                                              \
            throw std::runtime_error(                                  \
                std::string(__FILE__) + std::string(":") +             \
                std::to_string(__LINE__) + std::string(" in ") +       \
                std::string(__PRETTY_FUNCTION__) + std::string(": ") + \
                std::to_string((x)) + std::string(" != ") +            \
                std::to_string((y)));                                  \
        }                                                              \
    }

#define ASSERT_VECTOR_EQ(x, y)                                                               \
    {                                                                                        \
        for (int i = 0; i < x.size(); ++i)                                                   \
        {                                                                                    \
            ASSERT_EQ(x[i], y[i]);                                                           \
        }                                                                                    \
                                                                                             \
        if ((x.size()) != (y.size()))                                                        \
        {                                                                                    \
            throw std::runtime_error(                                                        \
                std::string(__FILE__) + std::string(":") +                                   \
                std::to_string(__LINE__) + std::string(" in ") +                             \
                std::string(__PRETTY_FUNCTION__) + std::string(": vector sizes not equal")); \
        }                                                                                    \
    }

int main()
{
    Matrix<double> mat1(2, 2, {1, 2, 3, 4});
    Matrix<double> mat2(2, 2, {8, 7, 6, 5});

    // test m-m addition
    std::vector<int> expected_mm_sum{9, 9, 9, 9};
    ASSERT_VECTOR_EQ((mat1 + mat2).v, expected_mm_sum);

    // test m-s addition
    std::vector<int> expected_ms_sum{12, 13, 14, 15};
    ASSERT_VECTOR_EQ((mat1 + 11).v, expected_ms_sum);

    // test s-m addition
    std::vector<int> expected_sm_sum{17, 16, 15, 14};
    ASSERT_VECTOR_EQ((9 + mat2).v, expected_sm_sum);

    return 0;
}
