#include "matrix.h"

template <typename T>
struct MultivariateNormal
{
    Matrix<T> mu;
    Matrix<T> sigma;

    template <typename Y = Matrix<T>>
    MultivariateNormal(Y &&M, Y &&S) : mu(std::forward<Y>(M)), sigma(std::forward<Y>(S))
    {
        if (M.n != 1)
            throw std::invalid_argument("M must be column vector");

        if (S.m != M.m || S.n != M.m)
            throw std::invalid_argument("S must be covariance matrix w/ dimensions N x N matching M");
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const MultivariateNormal<T> &mvn)
{
    std::cout << "Mean\n"
              << mvn.mu << "\nCovariance Matrix\n"
              << mvn.sigma << std::endl;
    return os;
}
