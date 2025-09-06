#pragma once

#include <random>
#include <vector>
#include <algorithm>

namespace MCL::RL::util
{
    template <typename T>
    class ReplayBuffer
    {
    public:
        inline static const size_t defaultCapacity = 10000;

    protected:
        std::vector<T> buffer;
        size_t capacity;
        size_t size;
        size_t nextIndex;

        mutable std::mt19937 gen;

    public:
        ReplayBuffer();
        ReplayBuffer(size_t capacity);

        void push(T);
        std::vector<T> getBatch(size_t) const;
        std::vector<std::vector<T>> getBatches(size_t) const;

        size_t getSize() const;
        size_t getCapacity() const;
        void resizeCapacity(size_t);
    };

    template <typename T>
    ReplayBuffer<T>::ReplayBuffer() : ReplayBuffer(defaultCapacity) {}

    template <typename T>
    ReplayBuffer<T>::ReplayBuffer(size_t capacity)
        : capacity(capacity), size(0), nextIndex(0), buffer(capacity), gen(std::random_device{}()) {}

    template <typename T>
    void ReplayBuffer<T>::push(T t)
    {
        buffer[nextIndex] = t;
        nextIndex = (nextIndex + 1) % capacity;
        if (size < capacity)
            ++size;
    }

    template <typename T>
    std::vector<T> ReplayBuffer<T>::getBatch(size_t batchsize) const
    {
        if (size < batchsize)
            batchsize = size;

        if (batchsize == 0)
            return std::vector<T>();

        std::vector<size_t> indices(size);
        size_t i;
        for (i = 0; i < size; ++i)
        {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<T> batch(batchsize);

        for (i = 0; i < batchsize; ++i)
        {
            batch[i] = buffer[indices[i]];
        }

        return batch;
    }

    template <typename T>
    std::vector<std::vector<T>> ReplayBuffer<T>::getBatches(size_t batchsize) const
    {
        if (batchsize == 0)
            return std::vector<std::vector<T>>();

        std::vector<size_t> indices(size);

        size_t i;
        for (i = 0; i < size; ++i)
        {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), gen);

        size_t noBatches = size % batchsize == 0 ? size / batchsize : size / batchsize + 1;

        std::vector<std::vector<T>> ret(noBatches);
        std::vector<T> batch(batchsize);

        size_t start, j, currentbatchsize;
        i = 0;
        for (start = 0; start < size; start += batchsize)
        {
            currentbatchsize = std::min(batchsize, size - start);
            ret[i] = std::vector<T>(currentbatchsize);

            for (j = 0; j < currentbatchsize; ++j)
            {
                ret[i][j] = buffer[indices[start + j]];
            }
            ++i;
        }

        return ret;
    }

    template <typename T>
    size_t ReplayBuffer<T>::getSize() const { return size; }

    template <typename T>
    size_t ReplayBuffer<T>::getCapacity() const { return capacity; }

    template <typename T>
    void ReplayBuffer<T>::resizeCapacity(size_t newCapacity)
    {
        std::vector<T> rewinded(size);

        size_t i, i_origin = nextIndex;
        for (i = 0; i < size; ++i)
        {
            rewinded[i] = buffer[i_origin];
            i_origin = (i_origin + 1) % size;
        }

        buffer = rewinded;
        buffer.resize(newCapacity);
        capacity = newCapacity;
    }
}