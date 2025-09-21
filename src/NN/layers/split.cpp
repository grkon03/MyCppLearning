#include "split.hpp"

namespace MCL::NN::Layers
{
    //
    // SplitLayer
    //

    SplitLayer::SplitLayer() : SplitLayer(0) {}
    SplitLayer::SplitLayer(size_t noLines) { reset(noLines); }
    SplitLayer::SplitLayer(const SplitLayer &s)
        : lines(s.noLines), noLines(s.noLines), inputsize(s.inputsize), outputsize(s.outputsize), eachOutputsizes(s.eachOutputsizes)
    {
        size_t i, j;

        for (i = 0; i < noLines; ++i)
        {
            for (j = 0; j < s.lines[i].length(); ++j)
            {
                lines[i].addLayer(s.lines[i][j]);
            }
        }
    }

    size_t SplitLayer::inputSize() const { return inputsize; }
    size_t SplitLayer::outputSize() const { return outputsize; }

    math::Rmatrix SplitLayer::forward(const math::Rmatrix &input)
    {
        assert(input.noRows() == inputsize);

        size_t i, j;
        std::vector<math::Rmatrix> outputs(noLines);

        for (i = 0; i < noLines; ++i)
        {
            outputs[i] = lines[i].forward(input);
        }

        return math::Rmatrix::connectVertical(outputs);
    }

    math::Rmatrix SplitLayer::backward(const math::Rmatrix &gradOutput)
    {
        assert(gradOutput.noRows() == outputsize);

        auto grads = gradOutput.splitRows(eachOutputsizes);
        math::Rmatrix gradHere(inputsize, gradOutput.noColumns(), 0);

        size_t i;
        for (i = 0; i < noLines; ++i)
        {
            gradHere += lines[i].backward(grads[i]);
        }

        return gradHere;
    }

    std::unique_ptr<Layer> SplitLayer::copy() const
    {
        return std::unique_ptr<Layer>(new SplitLayer(*this));
    }

    std::vector<math::Rmatrix *> SplitLayer::getParameterRefs()
    {
        std::vector<math::Rmatrix *> allparams;

        size_t i, j;
        for (i = 0; i < noLines; ++i)
        {
            auto params = lines[i].getParameterRefs();
            for (j = 0; j < params.size(); ++j)
            {
                allparams.push_back(params[j]);
            }
        }

        return allparams;
    }

    std::vector<math::Rmatrix> SplitLayer::getGradients() const
    {
        std::vector<math::Rmatrix> allgrads;

        size_t i, j;
        for (i = 0; i < noLines; ++i)
        {
            auto grads = lines[i].getGradients();
            for (j = 0; j < grads.size(); ++j)
            {
                allgrads.push_back(grads[j]);
            }
        }

        return allgrads;
    }

    void SplitLayer::reset(size_t _noLines)
    {
        noLines = _noLines;
        inputsize = 0;
        outputsize = 0;
        lines = std::vector<ConnectedLayers>(_noLines);
        eachOutputsizes = std::vector<size_t>(_noLines);
    }

    void SplitLayer::addLayer(size_t linenumber, const Layer *layer)
    {
        lines[linenumber].addLayer(layer);

        if (inputsize == 0)
            inputsize = layer->inputSize();
        else
            assert(inputsize == layer->inputSize());

        size_t i;
        outputsize = 0;
        for (i = 0; i < noLines; ++i)
        {
            outputsize += lines[i].outputSize();
            eachOutputsizes[i] = lines[i].outputSize();
        }
    }

    //
    // SplitLastLayer
    //

    SplitLastLayer::SplitLastLayer() : SplitLastLayer(0) {}
    SplitLastLayer::SplitLastLayer(size_t noSplits) { reset(noSplits); }
    SplitLastLayer::SplitLastLayer(const SplitLastLayer &sll)
        : inputsize(sll.inputsize), outputsize(sll.outputsize), eachInputsizes(sll.eachInputsizes), eachOutputsizes(sll.eachOutputsizes),
          noSplits(sll.noSplits), lastlayers(sll.noSplits), lossWeights(sll.lossWeights)
    {

        size_t i;
        for (i = 0; i < noSplits; ++i)
        {
            lastlayers[i] = std::move(sll.lastlayers[i]->copy());
        }
    }

    math::Rmatrix SplitLastLayer::forward(const math::Rmatrix &input)
    {
        auto splited = input.splitRows(eachInputsizes);
        std::vector<math::Rmatrix> predictions(noSplits);

        size_t i = 0;
        for (i = 0; i < noSplits; ++i)
        {
            predictions[i] = lastlayers[i]->forward(splited[i]);
        }

        _prediction = math::Rmatrix::connectVertical(predictions);

        _loss = 0;
        for (i = 0; i < noSplits; ++i)
        {
            _loss += lastlayers[i]->loss() * lossWeights[i];
        }

        return _prediction;
    }

    math::Rmatrix SplitLastLayer::backwardByComparing(const math::Rmatrix &compared)
    {
        assert(compared.noRows() == outputsize);
        auto splited = compared.splitRows(eachOutputsizes);
        std::vector<math::Rmatrix> grads(noSplits);

        size_t i;
        for (i = 0; i < noSplits; ++i)
        {
            grads[i] = lastlayers[i]->backwardByComparing(lossWeights[i] * splited[i]);
        }

        return math::Rmatrix::connectVertical(grads);
    }

    size_t SplitLastLayer::inputSize() const { return inputsize; }
    size_t SplitLastLayer::outputSize() const { return outputsize; }

    math::Rmatrix SplitLastLayer::prediction() const { return _prediction; }
    math::Real SplitLastLayer::loss() const { return _loss; }

    std::unique_ptr<LastLayer> SplitLastLayer::copy() const
    {
        return std::unique_ptr<LastLayer>(new SplitLastLayer(*this));
    }

    void SplitLastLayer::reset(size_t _noSplits)
    {
        noSplits = _noSplits;
        inputsize = 0;
        outputsize = 0;
        eachInputsizes = std::vector<size_t>(noSplits, 0);
        eachOutputsizes = std::vector<size_t>(noSplits, 0);
        lastlayers = std::move(std::vector<std::unique_ptr<LastLayer>>(noSplits));
        lossWeights = std::vector<math::Real>(noSplits, 0);
    }

    void SplitLastLayer::addLastLayer(size_t index, math::Real lossWeight, const LastLayer *ll)
    {
        lastlayers[index] = std::move(ll->copy());
        lossWeights[index] = lossWeight;
        eachInputsizes[index] = ll->inputSize();
        eachOutputsizes[index] = ll->outputSize();

        size_t i;
        inputsize = 0;
        outputsize = 0;
        for (i = 0; i < noSplits; ++i)
        {
            inputsize += eachInputsizes[i];
            outputsize += eachOutputsizes[i];
        }
    }
}