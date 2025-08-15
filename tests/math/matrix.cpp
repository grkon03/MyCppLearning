#include <mcl.hpp>
#include <gtest/gtest.h>
#include <random>

using namespace MCL;

TEST(MCLmath, MatrixBlockConstructor)
{
    math::Rmatrix m1({{1, 2}, {3, 4}}), m2({{5, 6}, {7, 8}}), m3({{9, 10}}), m4({{11, 12}});
    math::Rmatrix expect({{m1, m2}, {m3, m4}});
    math::Rmatrix correct({{1, 2, 5, 6}, {3, 4, 7, 8}, {9, 10, 11, 12}});
    EXPECT_EQ(expect, correct);
}

TEST(MCLmath, MatrixPadding)
{
    math::Rmatrix m({{1, 2}, {3, 4}});
    math::Rmatrix expect = m.padding(2, 1, 1, 2);
    math::Rmatrix correct({
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 1, 2, 0, 0},
        {0, 3, 4, 0, 0},
        {0, 0, 0, 0, 0},
    });

    EXPECT_EQ(expect, correct);
}

TEST(MCLmath, MatrixSubmatrix)
{
    math::Rmatrix m({{1, 2, 3, 4, 5},
                     {6, 7, 8, 9, 10},
                     {11, 12, 13, 14, 15},
                     {16, 17, 18, 19, 20}});
    math::Rmatrix expect = m.submatrix(1, 2, 2, 3);
    math::Rmatrix correct({{8, 9, 10}, {13, 14, 15}});

    EXPECT_EQ(expect, correct);
}

TEST(MCLmath, MatrixStrassen)
{
    const size_t M = 1000, N = 1050, K = 950;
    const size_t seed = 1000;
    math::Rmatrix m1(M, N), m2(N, K);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(-10, 10);

    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            m1.at(i, j) = dist(gen);
        }
    }

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < K; ++j)
        {
            m2.at(i, j) = dist(gen);
        }
    }

    math::Rmatrix::setStrassenSize(1500);
    math::Rmatrix correct = m1 * m2;
    math::Rmatrix::setStrassenSize(500);
    math::Rmatrix expect = m1 * m2;
    math::Rmatrix::setStrassenSizeDefault();

    EXPECT_EQ(expect, correct);
}