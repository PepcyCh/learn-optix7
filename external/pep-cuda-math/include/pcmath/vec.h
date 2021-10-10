#pragma once

#include <cmath>

#include "cuda_macro_utils.h"

namespace pep::cuda_math {

namespace details {

template <typename T, size_t N>
class VecT{
public:
    VecT() = default;

    CUDA_HOST_DEVICE VecT(std::initializer_list<T> &&list) {
        size_t i = 0;
        for (T val : list) {
            data_[i++] = val;
        }
        for (; i < N; i++) {
            data_[i] = static_cast<T>(0);
        }
    }

    template <typename... Args, std::enable_if_t<sizeof...(Args) == N, int> = 0>
    CUDA_HOST_DEVICE explicit VecT(Args... args) : VecT({ static_cast<T>(args)... }) {}

    template <size_t M>
    CUDA_HOST_DEVICE explicit VecT(const VecT<T, M> &rhs) {
        if constexpr (M < N) {
            for (size_t i = 0; i < M; i++) {
                data_[i] = rhs[i];
            }
            for (size_t i = M; i < N; i++) {
                data_[i] = static_cast<T>(0);
            }
        } else {
            for (size_t i = 0; i < N; i++) {
                data_[i] = rhs[i];
            }
        }
    }

    template <size_t M, typename... Args, std::enable_if_t<sizeof...(Args) + M <= N, int> = 0>
    CUDA_HOST_DEVICE explicit VecT(const VecT<T, M> &rhs, Args... args) {
        for (size_t i = 0; i < M; i++) {
            data_[i] = rhs[i];
        }
        const int len = sizeof...(args);
        const T last[] = { static_cast<T>(args)... };
        for (size_t i = 0; i < len; i++) {
            data_[M + i] = last[i];
        }
        for (size_t i = M + len; i < N; i++) {
            data_[i] = static_cast<T>(0);
        }
    }

    static CUDA_HOST_DEVICE VecT<T, N> Zero() {
        return VecT<T, N>({ static_cast<T>(0) });
    }
    static CUDA_HOST_DEVICE VecT<T, N> UnitX() {
        return VecT<T, N>({ static_cast<T>(1) });
    }
    static CUDA_HOST_DEVICE VecT<T, N> UnitY() {
        return VecT<T, N>({ static_cast<T>(0), static_cast<T>(1) });
    }
    static CUDA_HOST_DEVICE VecT<T, N> UnitZ() {
        static_assert(N >= 3, "Vector with less than 3 elements doesn't have field Z");
        return VecT<T, N>({ static_cast<T>(0), static_cast<T>(0), static_cast<T>(1) });
    }
    static CUDA_HOST_DEVICE VecT<T, N> UnitW() {
        static_assert(N >= 4, "Vector with less than 4 elements doesn't have field W");
        return VecT<T, N>({ static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1) });
    }

    CUDA_HOST_DEVICE T operator[](size_t index) const {
        return data_[index];
    }
    CUDA_HOST_DEVICE T &operator[](size_t index) {
        return data_[index];
    }

    CUDA_HOST_DEVICE T X() const {
        return this->data_[0];
    }
    CUDA_HOST_DEVICE T &X() {
        return this->data_[0];
    }

    CUDA_HOST_DEVICE T Y() const {
        return this->data_[1];
    }
    CUDA_HOST_DEVICE T &Y() {
        return this->data_[1];
    }

    CUDA_HOST_DEVICE T Z() const {
        static_assert(N >= 3, "Vector with less than 3 elements doesn't have field Z");
        return this->data_[2];
    }
    CUDA_HOST_DEVICE T &Z() {
        static_assert(N >= 3, "Vector with less than 3 elements doesn't have field Z");
        return this->data_[2];
    }

    CUDA_HOST_DEVICE T W() const {
        static_assert(N >= 4, "Vector with less than 4 elements doesn't have field W");
        return this->data_[3];
    }
    CUDA_HOST_DEVICE T &W() {
        static_assert(N >= 4, "Vector with less than 4 elements doesn't have field W");
        return this->data_[3];
    }

    CUDA_HOST_DEVICE VecT<T, N> operator+(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] + rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator+=(const VecT<T, N> &rhs) {
        for (size_t i = 0; i < N; i++) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator-() const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = -data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator-(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] - rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator-=(const VecT<T, N> &rhs) {
        for (size_t i = 0; i < N; i++) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator*(T rhs) const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] * rhs;
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator*=(T rhs) {
        for (size_t i = 0; i < N; i++) {
            data_[i] *= rhs;
        }
        return *this;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator*(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] * rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator*=(const VecT<T, N> &rhs) {
        for (size_t i = 0; i < N; i++) {
            data_[i] *= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator/(T rhs) const {
        VecT<T, N> res;
        const T inv = static_cast<T>(1) / rhs;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] * inv;
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator/=(T rhs) {
        const T inv = static_cast<T>(1) / rhs;
        for (size_t i = 0; i < N; i++) {
            data_[i] *= inv;
        }
        return *this;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator/(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (size_t i = 0; i < N; i++) {
            res.data_[i] = data_[i] / rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator/=(const VecT<T, N> &rhs) {
        for (size_t i = 0; i < N; i++) {
            data_[i] /= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE T Dot(const VecT<T, N> &rhs) const {
        T res = static_cast<T>(0);
        for (size_t i = 0; i < N; i++) {
            res += data_[i] * rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE T MagnitudeSqr() const {
        T res = static_cast<T>(0);
        for (size_t i = 0; i < N; i++) {
            res += data_[i] * data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE T Magnitude() const {
        return sqrt(MagnitudeSqr());
    }
    CUDA_HOST_DEVICE T Length() const {
        return sqrt(MagnitudeSqr());
    }

    CUDA_HOST_DEVICE VecT<T, N> Normalize() const {
        T inv = static_cast<T>(1) / Length();
        return *this * inv;
    }

    CUDA_HOST_DEVICE VecT<T, 3> Cross(const VecT<T, 3> &rhs) const {
        static_assert(N == 3, "Only vector with 3 elements can do cross product");
        VecT<T, 3> res(
            Y() * rhs.Z() - Z() * rhs.Y(),
            Z() * rhs.X() - X() * rhs.Z(),
            X() * rhs.Y() - Y() * rhs.X()
        );
        return res;
    }

    template <typename S, size_t M>
    friend class VecT;

private:
    T data_[N];
};

template <typename T, size_t N>
CUDA_HOST_DEVICE VecT<T, N> operator*(T lhs, const VecT<T, N> &rhs) {
    return rhs * lhs;
}

}

using Vec2 = details::VecT<float, 2>;
using IVec2 = details::VecT<int, 2>;
using DVec2 = details::VecT<double, 2>;
using BVec2 = details::VecT<bool, 2>;

using Vec3 = details::VecT<float, 3>;
using IVec3 = details::VecT<int, 3>;
using DVec3 = details::VecT<double, 3>;
using BVec3 = details::VecT<bool, 3>;

using Vec4 = details::VecT<float, 4>;
using IVec4 = details::VecT<int, 4>;
using DVec4 = details::VecT<double, 4>;
using BVec4 = details::VecT<bool, 4>;

}