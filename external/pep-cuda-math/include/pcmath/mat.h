#pragma once

#include "vec.h"

namespace pep::cuda_math {

namespace details {

template <typename T, size_t R, size_t C>
class MatT {
public:
    MatT() = default;

    CUDA_HOST_DEVICE MatT(std::initializer_list<VecT<T, R>> &&list) {
        size_t i = 0;
        for (const VecT<T, R> &val : list) {
            cols_[i++] = val;
        }
        for (; i < C; i++) {
            cols_[i] = VecT<T, R>::Zero();
        }
    }

    template <typename... Args, std::enable_if_t<sizeof...(Args) == R, int> = 0>
    CUDA_HOST_DEVICE explicit MatT(Args... args) : MatT({ static_cast<VecT<T, R>>(args)... }) {}

    template <size_t R2, size_t C2>
    CUDA_HOST_DEVICE explicit  MatT(const MatT<T, R2, C2> &rhs) {
        if constexpr (C2 < C) {
            for (size_t i = 0; i < C2; i++) {
                cols_[i] = VecT<T, R>(rhs[i]);
            }
            for (size_t i = C2; i < C; i++) {
                cols_[i] = VecT<T, R>::Zero();
            }
        } else {
            for (size_t i = 0; i < C; i++) {
                cols_[i] = VecT<T, R>(rhs[i]);
            }
        }
    }

    static CUDA_HOST_DEVICE MatT<T, R, C> Zero() {
        MatT<T, R, C> res({});
        return res;
    }
    static CUDA_HOST_DEVICE MatT<T, R, C> Identity() {
        static_assert(R == C, "Call Identity() to get a non square matrix");
        MatT<T, R, C> res = Zero();
        for (size_t i = 0; i < R; i++) {
            res[i][i] = static_cast<T>(1);
        }
        return res;
    }

    CUDA_HOST_DEVICE VecT<T, R> operator[](size_t index) const {
        return cols_[index];
    }
    CUDA_HOST_DEVICE VecT<T, R> &operator[](size_t index) {
        return cols_[index];
    }

    CUDA_HOST_DEVICE MatT<T, R, C> operator+(const MatT<T, R, C> &rhs) const {
        MatT<T, R, C> res;
        for (size_t i = 0; i < C; i++) {
            res.cols_[i] = cols_[i] + rhs.cols_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE MatT<T, R, C> &operator+=(const MatT<T, R, C> &rhs) {
        for (size_t i = 0; i < C; i++) {
            cols_[i] += rhs.cols_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE MatT<T, R, C> operator-() const {
        MatT<T, R, C> res;
        for (size_t i = 0; i < C; i++) {
            res.cols_[i] = -cols_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE MatT<T, R, C> operator-(const MatT<T, R, C> &rhs) const {
        MatT<T, R, C> res;
        for (size_t i = 0; i < C; i++) {
            res.cols_[i] = cols_[i] - rhs.cols_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE MatT<T, R, C> &operator-=(const MatT<T, R, C> &rhs) {
        for (size_t i = 0; i < C; i++) {
            cols_[i] -= rhs.cols_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE MatT<T, R, C> operator*(T rhs) const {
        MatT<T, R, C> res;
        for (size_t i = 0; i < R; i++) {
            res.cols_[i] = cols_[i] * rhs;
        }
        return res;
    }
    CUDA_HOST_DEVICE MatT<T, R, C> &operator*=(T rhs) {
        for (size_t i = 0; i < R; i++) {
            cols_[i] *= rhs;
        }
        return *this;
    }

    CUDA_HOST_DEVICE MatT<T, R, C> operator/(T rhs) const {
        MatT<T, R, C> res;
        const T inv = static_cast<T>(1) / rhs;
        for (size_t i = 0; i < R; i++) {
            res.cols_[i] = cols_[i] * inv;
        }
        return res;
    }
    CUDA_HOST_DEVICE MatT<T, R, C> &operator/=(T rhs) {
        const T inv = static_cast<T>(1) / rhs;
        for (size_t i = 0; i < R; i++) {
            cols_[i] *= inv;
        }
        return *this;
    }

    template <size_t M>
    MatT<T, R, M> operator*(const MatT<T, C, M> &rhs) const {
        MatT<T, R, M> res = Zero();
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < C; j++) {
                for (size_t k = 0; k < R; k++) {
                    res[i][k] += cols_[j][k] * rhs[i][j];
                }
            }
        }
        return res;
    }
    template <size_t M>
    MatT<T, R, M> &operator*=(const MatT<T, C, M> &rhs) {
        return *this = *this * rhs;
    }

    VecT<T, R> operator*(const VecT<T, C> &rhs) const {
        VecT<T, R> res = VecT<T, R>::Zero();
        for (size_t i = 0; i < C; i++) {
            res += cols_[i] * rhs[i];
        }
        return res;
    }

    MatT<T, C, R> Transpose() const {
        MatT<T, C, R> res;
        for (size_t i = 0; i < R; i++) {
            for (size_t j = 0; j < C; j++) {
                res[i][j] = cols_[j][i];
            }
        }
        return res;
    }

    MatT<T, R, C> Inverse() const {
        static_assert(R == C, "Call Inverse() on non square matrix");
        MatT<T, R, C> res = MatT::Identity();
        MatT<T, R, C> temp = *this;
        for (size_t i = 0; i < C; i++) {
            size_t pivot = i;
            T max = std::abs(temp[i][i]);
            for (size_t j = i + 1; j < R; j++) {
                T val = std::abs(temp[i][j]);
                if (val > max) {
                    max = val;
                    pivot = j;
                }
            }

            if (i != pivot) {
                for (size_t j = 0; j < R; j++) {
                    std::swap(temp[j][i], temp[j][pivot]);
                    std::swap(res[j][i], res[j][pivot]);
                }
            }

            if (max < static_cast<T>(0.000001)) {
                return res;
            }
            const T pivot_inv = static_cast<T>(1) / max;
            for (size_t j = 0; j < R; j++) {
                if (i != j) {
                    const T fact = temp[i][j] * pivot_inv;
                    for (size_t k = 0; k < C; k++) {
                        temp[k][j] -= fact * temp[k][i];
                        res[k][j] -= fact * res[k][i];
                    }
                }
            }

            for (size_t j = 0; j < R; j++) {
                temp[j][i] *= pivot_inv;
                res[j][i] *= pivot_inv;
            }
        }

        return res;
    }

    template <typename S, size_t R2, size_t C2>
    friend class MatT;

private:
    VecT<T, R> cols_[C];
};

template <typename T, size_t R, size_t C>
CUDA_HOST_DEVICE MatT<T, R, C> operator*(T lhs, const MatT<T, R, C> &rhs) {
    return rhs * lhs;
}

}

using Mat2 = details::MatT<float, 2, 2>;
using IMat2 = details::MatT<int, 2, 2>;
using DMat2 = details::MatT<double, 2, 2>;
using BMat2 = details::MatT<bool, 2, 2>;

using Mat3 = details::MatT<float, 3, 3>;
using IMat3 = details::MatT<int, 3, 3>;
using DMat3 = details::MatT<double, 3, 3>;
using BMat3 = details::MatT<bool, 3, 3>;

using Mat4 = details::MatT<float, 4, 4>;
using IMat4 = details::MatT<int, 4, 4>;
using DMat4 = details::MatT<double, 4, 4>;
using BMat4 = details::MatT<bool, 4, 4>;

}