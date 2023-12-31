#include <type_traits>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <vector>

template <typename T>
struct Matrix
{
    static_assert(std::is_floating_point<T>::value, "T must be floating point");

    std::vector<T> v;
    typename std::vector<T>::size_type m;
    typename std::vector<T>::size_type n;

    // TODO: check for overflow on M * N
    Matrix(typename std::vector<T>::size_type M, typename std::vector<T>::size_type N) : v(M * N), m(M), n(N)
    {
        if (M == 0 || N == 0)
            throw std::invalid_argument("Matrix dimension of 0 is undefined behvaior");
    }

    template <typename Y = std::vector<T>>
    Matrix(typename std::vector<T>::size_type M, typename std::vector<T>::size_type N, Y &&V) : v(std::forward<Y>(V)), m(M), n(N)
    {
        if (M == 0 || N == 0)
            throw std::invalid_argument("Matrix dimension of 0 is undefined behvaior");

        if (M * N != v.size())
            throw std::invalid_argument("Length of vector must equal product of matrix dimensions");
    }

    T get(typename std::vector<T>::size_type mi, typename std::vector<T>::size_type ni) const
    {
        if (mi >= m || ni >= n)
            throw std::invalid_argument("Row and col of matrix must be within bounds (0-indexed)");

        return v[n * mi + ni];
    }

    void set(typename std::vector<T>::size_type mi, typename std::vector<T>::size_type ni, T x)
    {
        if (mi >= m || ni >= n)
            throw std::invalid_argument("Row and col of matrix must be within bounds (0-indexed)");

        v[n * mi + ni] = x;
    }

    // transpose of matrix
    Matrix<T> t() const noexcept
    {
        Matrix<T> out(n, m);

        for (typename std::vector<T>::size_type i = 0; i < m; ++i)
            for (typename std::vector<T>::size_type j = 0; j < n; ++j)
                out.set(j, i, get(i, j));

        return out;
    }

    // TODO: currently uses Gaussian Elimination w/ time complexity O(n^3)
    // Consider more efficient LU/QR decomposition algorithms if this becomes a bottleneck
    Matrix<T> inv() const
    {
        if (m != n)
            throw std::invalid_argument("Matrix must be square");

        Matrix<T> mat = *this;
        Matrix<T> augmented = eye(n);

        // iterate over pivot row + column
        for (typename std::vector<T>::size_type p = 0; p < n; ++p)
        {
            T col_max_abs_val = 0;
            typename std::vector<T>::size_type i_row_max = 0;

            // find the k-th pivot
            for (typename std::vector<T>::size_type i = p; i < n; ++i)
            {
                T col_abs_val = std::abs(mat.get(i, p));

                if (col_abs_val > col_max_abs_val)
                {
                    col_max_abs_val = col_abs_val;
                    i_row_max = i;
                }
            }

            if (!(col_max_abs_val > 0))
                throw std::invalid_argument("Matrix is not invertible");

            mat.swapRows(p, i_row_max);
            augmented.swapRows(p, i_row_max);

            assert(std::abs(mat.get(p, p)) > 0);
            T f = 1.0 / mat.get(p, p);

            mat.scaleRow(p, f);
            augmented.scaleRow(p, f);

            // do for all rows in pivot column
            for (typename std::vector<T>::size_type i = 0; i < n; ++i)
            {
                if (i == p)
                    continue;

                T ftmp = mat.get(i, p);

                for (typename std::vector<T>::size_type j = 0; j < n; ++j)
                {
                    mat.set(i, j, mat.get(i, j) - mat.get(p, j) * ftmp);
                    augmented.set(i, j, augmented.get(i, j) - augmented.get(p, j) * ftmp);
                }
            }
        }

        return augmented;
    }

    // standard elementary row operation in gaussian elimination
    void scaleRow(typename std::vector<T>::size_type r, T scale)
    {
        if (r >= m)
            throw std::invalid_argument("Row must be within bounds of m (0-indexed)");

        for (typename std::vector<T>::size_type ni = 0; ni < n; ++ni)
            v[m * r + ni] *= scale;
    }

    // standard elementary row operation in gaussian elimination
    void swapRows(typename std::vector<T>::size_type r1, typename std::vector<T>::size_type r2)
    {
        if (r1 >= m || r2 >= m)
            throw std::invalid_argument("Rows must be within bounds of m (0-indexed)");

        for (typename std::vector<T>::size_type ni = 0; ni < n; ++ni)
            std::swap(v[n * r1 + ni], v[n * r2 + ni]);
    }

    // 1 x 1 scalar matrix
    static Matrix<T> scalar(T x) noexcept
    {
        Matrix<T> out(1, 1);
        out.v[0] = x;
        return out;
    }

    // N x N identity matrix
    static Matrix<T> eye(typename std::vector<T>::size_type n) noexcept
    {
        Matrix<T> out(n, n);

        for (typename std::vector<T>::size_type i = 0; i < n; ++i)
            out.set(i, i, 1);

        return out;
    }

    // N x 1 column vector
    static Matrix<T> col(typename std::vector<T>::size_type n, T x) noexcept
    {
        Matrix<T> out(n, 1);
        std::fill(out.v.begin(), out.v.end(), x);
        return out;
    }
};

// TODO: implement manageable optimizations for operations
// Cache + pre-fetching: https://lwn.net/Articles/255364/
// SIMD processing: https://codereview.stackexchange.com/questions/101144/simd-matrix-multiplication
// Sub-cubic multiplication algorithms: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm

template <typename T>
Matrix<T> operator+(const Matrix<T> &lhs, const Matrix<T> &rhs)
{
    if (lhs.m != rhs.m || lhs.n != rhs.n)
        throw std::invalid_argument("Matrices must be equal in dimensions");

    Matrix<T> out = lhs;

    for (typename std::vector<T>::size_type i = 0; i < out.v.size(); ++i)
        out.v[i] += rhs.v[i];

    return out;
}

template <typename T, typename C>
Matrix<T> operator+(const Matrix<T> &lhs, C rhs)
{
    static_assert(std::is_arithmetic<C>::value, "C must be numeric");

    Matrix<T> out = lhs;

    for (typename std::vector<T>::size_type i = 0; i < out.v.size(); ++i)
        out.v[i] += rhs;

    return out;
}

template <typename T, typename C>
Matrix<T> operator+(C lhs, const Matrix<T> &rhs) { return rhs + lhs; }

template <typename T>
Matrix<T> operator*(const Matrix<T> &lhs, const Matrix<T> &rhs)
{
    if (lhs.n != rhs.m)
        throw std::invalid_argument("Inner dimension of matrices must match");

    Matrix<T> out(lhs.m, rhs.n);

    for (typename std::vector<T>::size_type i = 0; i < out.m; ++i)
        for (typename std::vector<T>::size_type j = 0; j < out.n; ++j)
            for (typename std::vector<T>::size_type k = 0; k < lhs.n; ++k)
                out.set(i, j, out.get(i, j) + lhs.get(i, k) * rhs.get(k, j));

    return out;
}

template <typename T, typename C>
Matrix<T> operator*(const Matrix<T> &lhs, C rhs)
{
    static_assert(std::is_arithmetic<C>::value, "C must be numeric");

    Matrix<T> out = lhs;

    for (typename std::vector<T>::size_type i = 0; i < out.v.size(); ++i)
        out.v[i] *= rhs;

    return out;
}

template <typename T, typename C>
Matrix<T> operator*(C lhs, const Matrix<T> &rhs) { return rhs * lhs; }

template <typename T>
Matrix<T> operator-(const Matrix<T> &lhs, const Matrix<T> &rhs) { return lhs + (-1 * rhs); }

template <typename T, typename C>
Matrix<T> operator/(const Matrix<T> &lhs, C rhs)
{
    static_assert(std::is_arithmetic<C>::value, "C must be numeric");

    if (rhs == 0)
    {
        throw std::invalid_argument("Division by 0 is not defined");
    }

    return lhs * (1.0 / rhs);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &mat)
{
    for (typename std::vector<T>::size_type i = 0; i < mat.m; ++i)
    {
        for (typename std::vector<T>::size_type j = 0; j < mat.n; ++j)
            os << mat.get(i, j) << ' ';
        os << '\n';
    }

    return os;
}
