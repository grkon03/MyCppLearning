#pragma once

#include <tuple>
#include <blas.hh>
#include <iostream>
#include <cassert>
#include <vector>
#include <functional>
#include "arith.hpp"

namespace MCL::math
{
    template <arith T>
    class matrix
    {
    public:
        static inline size_t strassenSize = 10000;

        static inline void setStrassenSize(size_t size)
        {
            strassenSize = size;
        }
        static inline void setStrassenSizeDefault()
        {
            strassenSize = 10000;
        }

    private:
        size_t R;    // num of rows
        size_t C;    // num of columns
        size_t RC;   // R * C
        T *elements; // R x C elements

    public:
        matrix();
        matrix(size_t, size_t);
        matrix(size_t, size_t, T);
        matrix(std::initializer_list<std::initializer_list<T>>);
        matrix(std::initializer_list<std::initializer_list<matrix<T>>>);
        matrix(size_t, size_t, std::initializer_list<T>);
        matrix(const matrix<T> &);

        explicit matrix(T);

        ~matrix();

        // basic methods

        T &at(size_t, size_t);
        const T &at(size_t, size_t) const;

        T &direct(size_t);
        const T &direct(size_t) const;

        size_t noRows() const;
        size_t noColumns() const;

        matrix<T> submatrix(size_t rlocation, size_t clocation, size_t rsize, size_t csize) const;
        matrix<T> padding(size_t top, size_t bottom, size_t left, size_t right) const;

        bool isVVector() const;            // is a vertical vector
        bool isVVector(size_t size) const; // is vertical vector

        bool isHVector() const;            // is a horizontal vector
        bool isHVector(size_t size) const; // is a horizontal vector

        matrix<T> reshape(size_t rows, size_t cols) const;

        // operations

        matrix<T> operator-() const;

        matrix<T> operator+(matrix<T>) const;
        matrix<T> operator+(T) const;
        matrix<T> operator-(matrix<T>) const;
        matrix<T> operator-(T) const;
        matrix<T> operator*(matrix<T>) const;
        matrix<T> operator*(T) const;
        matrix<T> operator/(T) const;

        matrix<T> strassen(matrix<T>) const;

        const matrix<T> &operator+=(matrix<T>);
        const matrix<T> &operator+=(T);
        const matrix<T> &operator-=(matrix<T>);
        const matrix<T> &operator-=(T);
        const matrix<T> &operator*=(T);

        matrix<T> &operator=(const matrix<T> &mat);

        matrix<T> transpose() const;

        template <arith U = T>
        matrix<U> map(std::function<U(T)>) const;
        template <arith U = T>
        matrix<U> map(std::function<U(T, size_t)>) const;

        matrix<T> connectToTop(matrix<T>) const;
        matrix<T> connectToBottom(matrix<T>) const;
        matrix<T> connectToLeft(matrix<T>) const;
        matrix<T> connectToRight(matrix<T>) const;

        // comparations

        bool operator==(matrix<T>) const;
        bool operator==(T) const;

        bool isSameShape(matrix<T>) const;

        // arithmetical

        T sum() const;
        T max() const;
        std::pair<size_t, size_t> argmax() const;
        matrix<T> hadamardProd(matrix<T>) const;
    };

    template <arith T>
    matrix<T>::matrix() : R(0), C(0), RC(0) {}

    template <arith T>
    matrix<T>::matrix(size_t noRows, size_t noColumns)
        : R(noRows), C(noColumns), RC(noRows * noColumns), elements(new T[noRows * noColumns]) {}

    template <arith T>
    matrix<T>::matrix(size_t noRows, size_t noColumns, T init) : matrix(noRows, noColumns)
    {
        for (size_t i = 0; i < RC; ++i)
        {
            this->elements[i] = init;
        }
    }

    template <arith T>
    matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> elems)
    {
        R = elems.size();
        if (R == 0)
        {
            C = 0;
            return;
        }
        C = (*elems.begin()).size();
        RC = R * C;

        this->elements = new T[RC];

        size_t i, j;

        i = 0;
        for (const std::initializer_list<T> &row : elems)
        {
            j = 0;
            assert(row.size() == C);
            for (const T &e : row)
            {
                this->elements[i * C + j] = e;
                ++j;
            }
            ++i;
        }
    }

    template <arith T>
    matrix<T>::matrix(size_t noRows, size_t noColumns, std::initializer_list<T> elems) : matrix(noRows, noColumns)
    {
        assert(elems.size() == noRows * noColumns);
        const T *elemsarr = elems.begin();
        for (size_t i = 0; i < RC; ++i)
        {
            this->elements[i] = elemsarr[i];
        }
    }

    template <arith T>
    matrix<T>::matrix(std::initializer_list<std::initializer_list<matrix<T>>> blocks)
    {
        size_t _R, _C;
        std::vector<size_t> rowshape, colshape;

        _R = 0;
        _C = 0;
        for (const std::initializer_list<matrix<T>> &row : blocks)
        {
            rowshape.push_back(_R);
            _R += row.begin()->R;
        }
        for (const matrix<T> &column1 : *blocks.begin())
        {
            colshape.push_back(_C);
            _C += column1.C;
        }

        R = _R;
        C = _C;
        RC = R * C;
        elements = new T[RC];

        size_t i_this, i_block, i_loc, j_loc, margin, count;

        i_loc = 0;
        for (const std::initializer_list<matrix<T>> &row : blocks)
        {
            j_loc = 0;
            for (const matrix<T> &block : row)
            {
                margin = rowshape[i_loc] * C + colshape[j_loc];
                count = 0;
                i_this = margin;
                for (i_block = 0; i_block < block.RC; ++i_block)
                {
                    this->elements[i_this] = block.elements[i_block];
                    if (++count == block.C)
                    {
                        count = 0;
                        margin += C;
                        i_this = margin;
                    }
                    else
                    {
                        ++i_this;
                    }
                }
                ++j_loc;
            }
            ++i_loc;
        }
    }

    template <arith T>
    matrix<T>::matrix(const matrix<T> &mat) : matrix(mat.R, mat.C)
    {
        for (size_t i = 0; i < RC; ++i)
        {
            this->elements[i] = mat.elements[i];
        }
    }

    template <arith T>
    matrix<T>::matrix(T t) : matrix(1, 1, t) {}

    template <arith T>
    matrix<T>::~matrix()
    {
        delete[] elements;
        elements = nullptr;
    }

    template <arith T>
    T &matrix<T>::at(size_t i, size_t j)
    {
        return this->elements[i * C + j];
    }

    template <arith T>
    const T &matrix<T>::at(size_t i, size_t j) const
    {
        return this->elements[i * C + j];
    }

    template <arith T>
    T &matrix<T>::direct(size_t i) { return this->elements[i]; }

    template <arith T>
    const T &matrix<T>::direct(size_t i) const { return this->elements[i]; }

    template <arith T>
    size_t matrix<T>::noRows() const { return R; }
    template <arith T>
    size_t matrix<T>::noColumns() const { return C; }

    template <arith T>
    matrix<T> matrix<T>::submatrix(size_t rlocation, size_t clocation, size_t rsize, size_t csize) const
    {
        matrix<T> ret(rsize, csize);
        size_t i_ret, i_this, margin, count = 0;

        margin = rlocation * C + clocation;
        i_this = margin;
        for (i_ret = 0; i_ret < ret.RC; ++i_ret)
        {
            ret.elements[i_ret] = this->elements[i_this];
            if (++count == csize)
            {
                count = 0;
                margin += C;
                i_this = margin;
            }
            else
            {
                ++i_this;
            }
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::padding(size_t top, size_t bottom, size_t left, size_t right) const
    {
        matrix<T> ret(R + top + bottom, C + left + right);
        size_t i = 0, r, i_origin = 0, first;

        const size_t top_p = top * ret.C;
        for (; i < top_p; ++i)
        {
            ret.elements[i] = 0;
        }
        i_origin = 0;
        for (r = 0; r < R; ++r)
        {
            first = i;
            for (; i < first + left; ++i)
            {
                ret.elements[i] = 0;
            }

            first = i;
            for (; i < first + C; ++i)
            {
                ret.elements[i] = this->elements[i_origin];
                ++i_origin;
            }

            first = i;
            for (; i < first + right; ++i)
            {
                ret.elements[i] = 0;
            }
        }
        for (; i < ret.RC; ++i)
        {
            ret.elements[i] = 0;
        }

        return ret;
    }

    template <arith T>
    bool matrix<T>::isVVector() const
    {
        return C == 1;
    }

    template <arith T>
    bool matrix<T>::isVVector(size_t size) const
    {
        return (R == size && C == 1);
    }

    template <arith T>
    bool matrix<T>::isHVector() const
    {
        return R == 1;
    }

    template <arith T>
    bool matrix<T>::isHVector(size_t size) const
    {
        return (R == 1 && C == size);
    }

    template <arith T>
    matrix<T> matrix<T>::reshape(size_t rows, size_t cols) const
    {
        assert(rows * cols == RC);
        matrix<T> ret(rows, cols);
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i];
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator-() const
    {
        matrix<T> ret(R, C);
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = -this->elements[i];
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator+(matrix<T> mat) const
    {
        assert(R == mat.R && C == mat.C);
        matrix<T> ret(R, C);
        for (size_t i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] + mat.elements[i];
        }
        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator+(T a) const
    {
        matrix<T> ret(R, C);
        for (size_t i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] + a;
        }
        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator-(matrix<T> mat) const
    {
        assert(R == mat.R && C == mat.C);
        matrix<T> ret(R, C);

        for (size_t i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] - mat.elements[i];
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator-(T a) const
    {
        matrix<T> ret(R, C);
        for (size_t i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] - a;
        }
        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator*(matrix<T> mat) const
    {
        assert(C == mat.R);

        if (std::min({R, C, mat.C}) > strassenSize)
        {
            return strassen(mat);
        }

        matrix<T> ret(R, mat.C);

        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, ret.R, ret.C, C, 1.0,
                   this->elements, C, mat.elements, mat.C, 0.0, ret.elements, ret.C);

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator*(T t) const
    {
        matrix<T> ret(R, C);
        size_t i = 0;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] * t;
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::operator/(T t) const
    {
        matrix<T> ret(R, C);
        size_t i = 0;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] / t;
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::strassen(matrix<T> mat) const
    {
        matrix<T> A = this->padding(0, R % 2, 0, C % 2);
        matrix<T> B = mat.padding(0, mat.R % 2, 0, mat.C % 2);

        const size_t M2 = A.R / 2, N2 = A.C / 2, K2 = B.C / 2;

        matrix<T> A11, A12, A21, A22, B11, B12, B21, B22;
        A11 = A.submatrix(0, 0, M2, N2);
        A12 = A.submatrix(0, N2, M2, N2);
        A21 = A.submatrix(M2, 0, M2, N2);
        A22 = A.submatrix(M2, N2, M2, N2);
        B11 = B.submatrix(0, 0, N2, K2);
        B12 = B.submatrix(0, K2, N2, K2);
        B21 = B.submatrix(N2, 0, N2, K2);
        B22 = B.submatrix(N2, K2, N2, K2);

        matrix<T> P1, P2, P3, P4, P5, P6, P7;
        P1 = (A11 + A22) * (B11 + B22);
        P2 = (A21 + A22) * B11;
        P3 = A11 * (B12 - B22);
        P4 = A22 * (B21 - B11);
        P5 = (A11 + A12) * B22;
        P6 = (A21 - A11) * (B11 + B12);
        P7 = (A12 - A22) * (B21 + B22);

        matrix<T> C11, C12, C21, C22;
        C11 = P1 + P4 - P5 + P7;
        C12 = P3 + P5;
        C21 = P2 + P4;
        C22 = P1 - P2 + P3 + P6;

        return matrix<T>({{C11, C12}, {C21, C22}}).submatrix(0, 0, R, mat.C);
    }

    template <arith T>
    const matrix<T> &matrix<T>::operator+=(T t)
    {
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            this->elements[i] += t;
        }

        return *this;
    }

    template <arith T>
    const matrix<T> &matrix<T>::operator+=(matrix<T> mat)
    {
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            this->elements[i] += mat.elements[i];
        }

        return *this;
    }

    template <arith T>
    const matrix<T> &matrix<T>::operator-=(matrix<T> mat)
    {
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            this->elements[i] -= mat.elements[i];
        }

        return *this;
    }

    template <arith T>
    const matrix<T> &matrix<T>::operator-=(T t)
    {
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            this->elements[i] -= t;
        }

        return *this;
    }

    template <arith T>
    matrix<T> &matrix<T>::operator=(const matrix<T> &mat)
    {
        if (RC > 0)
            delete[] elements;
        size_t i;
        R = mat.R;
        C = mat.C;
        RC = mat.RC;
        elements = new T[RC];

        for (i = 0; i < RC; ++i)
        {
            this->elements[i] = mat.elements[i];
        }

        return *this;
    }

    template <arith T>
    matrix<T> matrix<T>::transpose() const
    {
        matrix<T> ret(C, R);

        size_t i, j;
        for (i = 0; i < C; ++i)
        {
            for (j = 0; j < R; ++j)
            {
                ret.elements[i * R + j] = this->elements[j * C + i];
            }
        }

        return ret;
    }

    template <arith T>
    template <arith U>
    matrix<U> matrix<T>::map(std::function<U(T)> func) const
    {
        matrix<U> ret(R, C);
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = func(this->elements[i]);
        }

        return ret;
    }

    template <arith T>
    template <arith U>
    matrix<U> matrix<T>::map(std::function<U(T, size_t)> func) const
    {
        matrix<U> ret(R, C);
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = func(this->elements[i], i);
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::connectToTop(matrix<T> mat) const
    {
        assert(C == mat.C);

        matrix<T> ret(R + mat.R, C);

        size_t i;
        for (i = 0; i < mat.RC; ++i)
        {
            ret.elements[i] = mat.elements[i];
        }
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i + mat.RC] = this->elements[i];
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::connectToBottom(matrix<T> mat) const
    {
        return mat.connectToTop(*this);
    }

    template <arith T>
    matrix<T> matrix<T>::connectToLeft(matrix<T> mat) const
    {
        assert(R == mat.R);

        matrix<T> ret(R, C + mat.C);

        size_t i, j;
        for (i = 0; i < R; ++i)
        {
            for (j = 0; j < mat.C; ++j)
            {
                ret.elements[i * ret.C + j] = mat.elements[i * mat.C + j];
            }
            for (j = 0; j < C; ++j)
            {
                ret.elements[i * ret.C + j + mat.C] = this->elements[i * C + j];
            }
        }

        return ret;
    }

    template <arith T>
    matrix<T> matrix<T>::connectToRight(matrix<T> mat) const
    {
        return mat.connectToLeft(*this);
    }

    template <arith T>
    bool matrix<T>::operator==(matrix<T> mat) const
    {
        if (C != mat.C || R != mat.R)
            return false;

        size_t i;
        for (i = 0; i < RC; ++i)
        {
            if (this->elements[i] != mat.elements[i])
                return false;
        }
        return true;
    }

    template <arith T>
    bool matrix<T>::isSameShape(matrix<T> mat) const
    {
        return (R == mat.R && C == mat.C);
    }

    template <arith T>
    T matrix<T>::sum() const
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            T sum = 0;
            T c = 0;
            for (size_t i = 0; i < RC; ++i)
            {
                T y = elements[i] - c;
                T t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            return sum;
        }
        else
        {
            T sum = 0;
            for (size_t i = 0; i < RC; ++i)
            {
                sum += elements[i];
            }
            return sum;
        }
    }

    template <arith T>
    T matrix<T>::max() const
    {
        static_assert(std::totally_ordered<T>, "max() can be used only in the case that T is totally ordered");

        assert(RC > 0);

        T _max = this->elements[0];

        size_t i;
        for (i = 1; i < RC; ++i)
        {
            if (_max < this->elements[i])
            {
                _max = this->elements[i];
            }
        }

        return _max;
    }

    template <arith T>
    std::pair<size_t, size_t> matrix<T>::argmax() const
    {
        static_assert(std::totally_ordered<T>, "argmax() can be used only in the case that T is totally ordered");

        assert(RC > 0);

        T _max = this->elements[0];
        size_t i, i_max = 0;
        for (i = 0; i < RC; ++i)
        {
            if (_max < this->elements[i])
            {
                _max = this->elements[i];
                i_max = i;
            }
        }

        return {i / C, i % C};
    }

    template <arith T>
    matrix<T> matrix<T>::hadamardProd(matrix<T> mat) const
    {
        assert(R == mat.R && C == mat.C);

        matrix<T> ret(R, C);
        size_t i;
        for (i = 0; i < RC; ++i)
        {
            ret.elements[i] = this->elements[i] * mat.elements[i];
        }

        return ret;
    }

    // other operators

    template <arith T>
    std::ostream &operator<<(std::ostream &os, const matrix<T> &mat)
    {
        size_t i, j;
        for (i = 0; i < mat.noRows(); ++i)
        {
            for (j = 0; j < mat.noColumns(); ++j)
            {
                os << mat.at(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

    template <arith T>
    matrix<T> operator+(T t, matrix<T> mat)
    {
        return mat + t;
    }

    template <arith T>
    matrix<T> operator-(T t, matrix<T> mat)
    {
        return -(mat - t);
    }

    template <arith T>
    matrix<T> operator*(T t, matrix<T> mat)
    {
        return mat * t;
    }

    using Rmatrix = matrix<Real>;
    using Imatrix = matrix<Integer>;
}