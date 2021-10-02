#pragma once

#include <cmath>

#include "cuda_macro_utils.h"

template<typename T, size_t N>
struct VecT {
    CUDA_HOST_DEVICE T operator[](size_t index) const {
        return data_[index];
    }

    CUDA_HOST_DEVICE VecT<T, N> operator+(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] + rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator+=(const VecT<T, N> &rhs) {
        for (int i = 0; i < N; i++) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator-() const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = -data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator-(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] - rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator-=(const VecT<T, N> &rhs) {
        for (int i = 0; i < N; i++) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator*(T rhs) const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] * rhs;
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator*=(T rhs) {
        for (int i = 0; i < N; i++) {
            data_[i] *= rhs;
        }
        return *this;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator*(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] * rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator*=(const VecT<T, N> &rhs) {
        for (int i = 0; i < N; i++) {
            data_[i] *= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE VecT<T, N> operator/(T rhs) const {
        VecT<T, N> res;
        T inv = static_cast<T>(1) / rhs;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] * inv;
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator/=(T rhs) {
        T inv = static_cast<T>(1) / rhs;
        for (int i = 0; i < N; i++) {
            data_[i] *= inv;
        }
        return *this;
    }
    CUDA_HOST_DEVICE VecT<T, N> operator/(const VecT<T, N> &rhs) const {
        VecT<T, N> res;
        for (int i = 0; i < N; i++) {
            res.data_[i] = data_[i] / rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE VecT<T, N> &operator/=(const VecT<T, N> &rhs) {
        for (int i = 0; i < N; i++) {
            data_[i] /= rhs.data_[i];
        }
        return *this;
    }

    CUDA_HOST_DEVICE T Dot(const VecT<T, N> &rhs) const {
        T res = static_cast<T>(0);
        for (int i = 0; i < N; i++) {
            res += data_[i] * rhs.data_[i];
        }
        return res;
    }
    CUDA_HOST_DEVICE T MagnitudeSqr() const {
        T res = static_cast<T>(0);
        for (int i = 0; i < N; i++) {
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

protected:
    VecT() = default;

    CUDA_HOST_DEVICE VecT(std::initializer_list<T> &&list) {
        int i = 0;
        for (T val : list) {
            data_[i++] = val;
        }
        for (; i < N; i++) {
            data_[i] = static_cast<T>(0);
        }
    }

    template<typename... Args, std::enable_if_t<sizeof...(Args) == N, int> = 0>
    CUDA_HOST_DEVICE VecT(Args... args) : VecT({static_cast<T>(args)...}) {}

    T data_[N];
};

template<typename T, size_t N>
CUDA_HOST_DEVICE VecT<T, N> operator*(T lhs, const VecT<T, N> &rhs) {
    return rhs * lhs;
}

template<typename T>
struct Vec2T : VecT<T, 2> {
    Vec2T() = default;
    CUDA_HOST_DEVICE Vec2T(const VecT<T, 2> &base) : VecT<T, 2>(base[0], base[1]) {}
    CUDA_HOST_DEVICE Vec2T(T x, T y) : VecT<T, 2>(x, y) {}

    CUDA_HOST_DEVICE T X() const {
        return this->data_[0];
    }

    CUDA_HOST_DEVICE T Y() const {
        return this->data_[1];
    }

    static const Vec2T<T> kZero;
    static const Vec2T<T> kX;
    static const Vec2T<T> kY;

#ifdef __CUDACC__
    static constexpr __device__ Vec2T<T> ZeroVec() {
        return Vec2T(static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec2T<T> UnitX() {
        return Vec2T(static_cast<T>(1), static_cast<T>(0));
    }
    static constexpr __device__ Vec2T<T> UnitY() {
        return Vec2T(static_cast<T>(0), static_cast<T>(1));
    }
#endif
};

template<typename T>
const Vec2T<T> Vec2T<T>::kZero = Vec2T(static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec2T<T> Vec2T<T>::kX = Vec2T(static_cast<T>(1), static_cast<T>(0));
template<typename T>
const Vec2T<T> Vec2T<T>::kY = Vec2T(static_cast<T>(0), static_cast<T>(1));

template<typename T>
struct Vec3T : VecT<T, 3> {
    Vec3T() = default;
    CUDA_HOST_DEVICE Vec3T(const VecT<T, 3> &base) : VecT<T, 3>(base[0], base[1], base[2]) {}
    CUDA_HOST_DEVICE Vec3T(T x, T y, T z) : VecT<T, 3>(x, y, z) {}
    CUDA_HOST_DEVICE Vec3T(const Vec2T<T> &xy, T z) : VecT<T, 3>(xy.X(), xy.Y(), z) {}

    CUDA_HOST_DEVICE T X() const {
        return this->data_[0];
    }

    CUDA_HOST_DEVICE T Y() const {
        return this->data_[1];
    }

    CUDA_HOST_DEVICE T Z() const {
        return this->data_[2];
    }

    static const Vec3T<T> kZero;
    static const Vec3T<T> kX;
    static const Vec3T<T> kY;
    static const Vec3T<T> kZ;

    CUDA_HOST_DEVICE Vec3T<T> Cross(const Vec3T<T> &rhs) const {
        Vec3T<T> res(
            Y() * rhs.Z() - Z() * rhs.Y(),
            Z() * rhs.X() - X() * rhs.Z(),
            X() * rhs.Y() - Y() * rhs.X()
        );
        return res;
    }

#ifdef __CUDACC__
    static constexpr __device__ Vec3T<T> ZeroVec() {
        return Vec3T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec3T<T> UnitX() {
        return Vec3T(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec3T<T> UnitY() {
        return Vec3T(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0));
    }
    static constexpr __device__ Vec3T<T> UnitZ() {
        return Vec3T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
    }
#endif
};

template<typename T>
const Vec3T<T> Vec3T<T>::kZero = Vec3T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec3T<T> Vec3T<T>::kX = Vec3T(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec3T<T> Vec3T<T>::kY = Vec3T(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0));
template<typename T>
const Vec3T<T> Vec3T<T>::kZ = Vec3T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));

template<typename T>
struct Vec4T : VecT<T, 4> {
    Vec4T() = default;
    CUDA_HOST_DEVICE Vec4T(const VecT<T, 4> &base) : VecT<T, 4>(base[0], base[1], base[2], base[3]) {}
    CUDA_HOST_DEVICE Vec4T(T x, T y, T z, T w) : VecT<T, 4>(x, y, z, w) {}
    CUDA_HOST_DEVICE Vec4T(const Vec2T<T> &xy, T z, T w) : VecT<T, 4>(xy.X(), xy.Y(), z, w) {}
    CUDA_HOST_DEVICE Vec4T(const Vec3T<T> &xyz, T w) : VecT<T, 4>(xyz.X(), xyz.Y(), xyz.Z(), w) {}

    CUDA_HOST_DEVICE T X() const {
        return this->data_[0];
    }

    CUDA_HOST_DEVICE T Y() const {
        return this->data_[1];
    }

    CUDA_HOST_DEVICE T Z() const {
        return this->data_[2];
    }

    CUDA_HOST_DEVICE T W() const {
        return this->data_[3];
    }

    static const Vec4T<T> kZero;
    static const Vec4T<T> kX;
    static const Vec4T<T> kY;
    static const Vec4T<T> kZ;
    static const Vec4T<T> kW;

#ifdef __CUDACC__
    static constexpr __device__ Vec4T<T> ZeroVec() {
        return Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec4T<T> UnitX() {
        return Vec4T(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec4T<T> UnitY() {
        return Vec4T(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
    }
    static constexpr __device__ Vec4T<T> UnitZ() {
        return Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0));
    }
    static constexpr __device__ Vec4T<T> UnitW() {
        return Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
    }
#endif
};

template<typename T>
const Vec4T<T> Vec4T<T>::kZero = Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec4T<T> Vec4T<T>::kX = Vec4T(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec4T<T> Vec4T<T>::kY = Vec4T(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0));
template<typename T>
const Vec4T<T> Vec4T<T>::kZ = Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(0));
template<typename T>
const Vec4T<T> Vec4T<T>::kW = Vec4T(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));

using Vec2 = Vec2T<float>;
using IVec2 = Vec2T<int>;
using DVec2 = Vec2T<double>;
using BVec2 = Vec2T<bool>;

using Vec3 = Vec3T<float>;
using IVec3 = Vec3T<int>;
using DVec3 = Vec3T<double>;
using BVec3 = Vec3T<bool>;

using Vec4 = Vec4T<float>;
using IVec4 = Vec4T<int>;
using DVec4 = Vec4T<double>;
using BVec4 = Vec4T<bool>;
