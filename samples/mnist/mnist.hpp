#pragma once

#include <cstdlib>

namespace MCLSamples::mnist
{
    struct Image
    {
        static const size_t rows = 28;
        static const size_t cols = 28;

        unsigned char data[rows * cols];
        int label;
    };

    struct Dataset
    {
        Image *images;
        size_t noImages;
    };

    Dataset loadTrain();
    Dataset loadTest();
}