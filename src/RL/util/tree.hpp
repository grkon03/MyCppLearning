#pragma once

#include <memory>
#include <vector>

namespace MCL::RL::util
{
    template <typename ValueType>
    class Node : public std::enable_shared_from_this<Node<ValueType>>
    {
    private:
        using _Node = Node<ValueType>;

        ValueType _value;
        std::vector<std::shared_ptr<_Node>> _children;
        std::weak_ptr<_Node> _parent;

    public:
        Node();
        Node(std::shared_ptr<_Node> parent);

        ValueType &value();
        const ValueType &value() const;

        std::shared_ptr<_Node> newchild();
        std::shared_ptr<_Node> child(size_t i);
        std::shared_ptr<_Node> parent();
        const std::vector<std::shared_ptr<_Node>> &children() const;

        void setChildren(const std::vector<std::shared_ptr<_Node>> &__children);
    };

    template <typename ValueType>
    Node<ValueType>::Node() {}
    template <typename ValueType>
    Node<ValueType>::Node(std::shared_ptr<_Node> _parent)
        : _parent(_parent) {}

    template <typename ValueType>
    ValueType &Node<ValueType>::value() { return _value; }

    template <typename ValueType>
    const ValueType &Node<ValueType>::value() const { return _value; }

    template <typename ValueType>
    std::shared_ptr<Node<ValueType>> Node<ValueType>::newchild()
    {
        return std::make_shared<_Node>(this->shared_from_this());
    }

    template <typename ValueType>
    std::shared_ptr<Node<ValueType>> Node<ValueType>::child(size_t i)
    {
        return _children.at(i);
    }

    template <typename ValueType>
    std::shared_ptr<Node<ValueType>> Node<ValueType>::parent()
    {
        return _parent.lock();
    }

    template <typename ValueType>
    const std::vector<std::shared_ptr<Node<ValueType>>> &Node<ValueType>::children() const
    {
        return _children;
    }

    template <typename ValueType>
    void Node<ValueType>::setChildren(const std::vector<std::shared_ptr<_Node>> &__children)
    {
        _children = __children;
    }

    template <typename ValueType>
    class Tree
    {
    public:
        using Node = Node<ValueType>;

    private:
        std::shared_ptr<Node> _root;

    public:
        Tree();

        std::shared_ptr<Node> root();
    };

    template <typename ValueType>
    Tree<ValueType>::Tree() : _root(std::make_shared<Node>()) {}

    template <typename ValueType>
    std::shared_ptr<Node<ValueType>> Tree<ValueType>::root()
    {
        return _root;
    }
}