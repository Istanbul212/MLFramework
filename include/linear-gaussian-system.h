#include "multivariate-normal.h"

template <typename T>
struct LinearGaussianSystem
{
    MultivariateNormal<T> posterior;

    Matrix<T> a;
    Matrix<T> b;

    template <typename Y = MultivariateNormal<T>, typename Q = Matrix<T>>
    LinearGaussianSystem(Y &&P, Q &&A, Q &&B) : posterior(std::forward<Y>(P)),
                                                a(std::forward<Q>(A)),
                                                b(std::forward<Q>(B)) {}

    void train(const MultivariateNormal<T> &y)
    {
        Matrix<T> sigma = (posterior.sigma.inv() + a.t() * y.sigma.inv() * a).inv();
        posterior.mu = sigma * (a.t() * y.sigma.inv() * (y.mu - b) + posterior.sigma.inv() * posterior.mu);
        posterior.sigma = sigma;
    }
};
