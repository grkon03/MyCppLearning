#pragma once

#include "layer.hpp"
#include "lastlayer.hpp"
#include "connected.hpp"

namespace MCL::NN::Layers
{
    /**
     * @brief
     *
     * an input will be propageted to each layers
     *
     */
    class SplitLayer : public Layer
    {
    private:
        size_t noLines;

        size_t inputsize;
        size_t outputsize;

        /**
         * @brief
         * Each lines[i] is a array of layers.
         * Each input go through like lines[i][j] -> lines[i][j + 1]
         */
        std::vector<ConnectedLayers> lines;

        std::vector<size_t> eachOutputsizes;

    public:
        SplitLayer();
        SplitLayer(size_t noLines);
        SplitLayer(const SplitLayer &);

        virtual size_t inputSize() const override;
        virtual size_t outputSize() const override;
        virtual math::Rmatrix forward(const math::Rmatrix &) override;  // data should be vertical vectors aligned horizontal as batches
        virtual math::Rmatrix backward(const math::Rmatrix &) override; // data should be vertical vectors aligned horizontal as batches

        virtual std::unique_ptr<Layer> copy() const override;

        // learn

        virtual std::vector<math::Rmatrix *> getParameterRefs() override;
        virtual std::vector<math::Rmatrix> getGradients() const override;

        void reset(size_t noLines);
        void addLayer(size_t linenumber, const Layer *);
    };

    class SplitLastLayer : public LastLayer
    {
    private:
        size_t noSplits;

        size_t inputsize;
        size_t outputsize;

        math::Rmatrix _prediction;
        math::Real _loss;

        std::vector<std::unique_ptr<LastLayer>> lastlayers;
        std::vector<math::Real> lossWeights;
        std::vector<size_t> eachInputsizes;
        std::vector<size_t> eachOutputsizes;

    public:
        SplitLastLayer();
        SplitLastLayer(size_t noSplits);
        SplitLastLayer(const SplitLastLayer &);

        virtual math::Rmatrix forward(const math::Rmatrix &) override;
        virtual math::Rmatrix backwardByComparing(const math::Rmatrix &compared) override;

        virtual size_t inputSize() const override;
        virtual size_t outputSize() const override;

        virtual math::Rmatrix prediction() const override;
        virtual math::Real loss() const override;

        virtual std::unique_ptr<LastLayer> copy() const override;

        void reset(size_t _noSplits);
        void addLastLayer(size_t index, math::Real lossWeight, const LastLayer *);
    };
}