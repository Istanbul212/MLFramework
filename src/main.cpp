#include <iostream>
#include "linear-gaussian-system.h"

int main()
{
    Matrix<double> tmp2(1, 3, {1, 2, 3});

    MultivariateNormal<double> mvn(Matrix<double>::col(2, 0), Matrix<double>::eye(2) * 1e10);

    LinearGaussianSystem<double> lgs(mvn,
                                     Matrix<double>::eye(2),
                                     Matrix<double>::col(2, 0));

    MultivariateNormal<double> y1(Matrix<double>(2, 1, {0, -1}), Matrix<double>(2, 2, {1, 0, 0, 1}) * 0.01);
    MultivariateNormal<double> y2(Matrix<double>(2, 1, {1, 0}), Matrix<double>(2, 2, {1, 0, 0, 1}) * 0.01);

    lgs.train(y1);
    lgs.train(y2);

    return 0;
}
