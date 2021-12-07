/** 
 * \file include/graphs/line.hpp
 *
 * Implementation of the line graph.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/7/2021
 */

#ifndef LINE_LABELED_DIGRAPH_HPP
#define LINE_LABELED_DIGRAPH_HPP

#include <sstream>
#include <cstdarg>
#include <vector>
#include <utility>
#include <stdexcept>
#include "../digraph.hpp"

/**
 * An implementation of the line graph.
 *
 * The nodes in the line graph are denoted by 0, ..., `this->N`, and therefore
 * we have `this->numnodes == this->N + 1`.
 *
 * Additional methods implemented for this graph include: 
 * - `getUpperExitProb()`: compute the *splitting probability* of exiting
 *   the graph through `this->N` (reaching an auxiliary upper exit node),
 *   and not through `0` (reaching an auxiliary lower exit node). 
 * - `getUpperExitRate(IOType)`: compute the reciprocal of the *unconditional
 *   mean first-passage time* to exiting the graph through `this->N`, given
 *   that the exit rate from `0` is zero. 
 * - `getLowerExitRate()`: compute the reciprocal of the *unconditional mean
 *   first-passage time* to exiting the graph through `0`, given that the 
 *   exit rate from `this->N` is zero. 
 * - `getUpperExitRate(IOType, IOType)`: compute the reciprocal of the *conditional
 *   mean first-passage time* to exiting the graph through `this->N`, given that
 *   (1) both exit rates are nonzero and (2) exit through `this->N` does occur. 
 */
template <typename InternalType, typename IOType = InternalType>
class LineGraph : public LabeledDigraph<InternalType, IOType>
{
    private:
        /**
         * Index of the last node, N; one less than the number of nodes. 
         */
        int N;

        /**
         * Vector of edge labels that grows with the length of the graph. 
         * The i-th pair stores the labels for the edges i -> i+1 and i+1 -> i.
         */
        std::vector<std::pair<InternalType, InternalType> > line_labels;

        /**
         * Add a node to the graph with the given ID, and return a pointer
         * to the new node.
         *
         * A convenient private version of the parent method
         * `LabeledDigraph<...>::addNode()`, so that any call to
         * `LineGraph<...>::addNode()` throws an exception.  
         *
         * @param id ID for new node. 
         * @returns  A pointer to new node. 
         * @throws std::runtime_error if node already exists with the given ID. 
         */
        Node* __addNode(std::string id)
        {
            // Check that a node with the given id doesn't exist already
            if (this->nodes.find(id) != this->nodes.end())
                throw std::runtime_error("Node already exists"); 

            Node* node = new Node(id);
            this->order.push_back(node); 
            this->nodes[id] = node;
            this->edges.emplace(node, std::unordered_map<Node*, InternalType>());
            this->numnodes++;
            return node;
        }

        /**
         * Add an edge between two nodes.
         *
         * If either ID does not correspond to a node in the graph, this function
         * instantiates these nodes.
         *
         * A convenient private version of the parent method
         * `LabeledDigraph<...>::addEdge()`, so that any call to
         * `LineGraph<...>::addEdge()` throws an exception.  
         *
         * @param source_id ID of source node of new edge. 
         * @param target_id ID of target node of new edge. 
         * @param label     Label on new edge.
         * @throws std::runtime_error if the edge already exists. 
         */
        void __addEdge(std::string source_id, std::string target_id, IOType label = 1)
        {
            // Look for the two nodes
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Then see if the edge already exists (in which case the two nodes
            // obviously already exist) 
            if (source != nullptr && target != nullptr &&
                this->edges[source].find(target) != this->edges[source].end())
            {
                throw std::runtime_error("Edge already exists; use LabeledDigraph<...>::setEdgeLabel() to change label");
            }

            // If the nodes do not exist, then add the nodes  
            if (source == nullptr)
                source = this->addNode(source_id);
            if (target == nullptr)
                target = this->addNode(target_id);

            // Then define the edge
            this->edges[source][target] = static_cast<InternalType>(label);
        }

    public:
        /**
         * Constructor for a line graph of length 1, i.e., a single vertex
         * named "0".
         */
        LineGraph() : LabeledDigraph<InternalType, IOType>()
        {
            this->__addNode("0");
            this->N = 0; 
        }

        /**
         * Constructor for a line graph of given length, with edge labels set
         * to unity.
         *
         * @param N Length of desired line graph; `this->N` is set to `N`, and 
         *          `this->numnodes` to `N + 1`. 
         */
        LineGraph(unsigned N) : LabeledDigraph<InternalType, IOType>()
        {
            // Add the zeroth node
            this->__addNode("0");

            for (unsigned i = 0; i < N; ++i)
            {
                // Add the (i+1)-th node
                std::stringstream ssi, ssj;
                ssi << i;
                ssj << i + 1;
                this->__addNode(ssj.str());

                // Add edges i -> i+1 and i+1 -> i with labels set to unity 
                this->__addEdge(ssi.str(), ssj.str());
                this->__addEdge(ssj.str(), ssi.str());

                // Separately keep track of edge labels
                std::pair<InternalType, InternalType> labels = {1, 1};
                this->line_labels.push_back(labels);
            }
            this->N = N; 
        }

        /**
         * Trivial destructor. 
         */
        ~LineGraph()
        {
        }

        /**
         * Add a new node to the end of the graph (along with the two edges),
         * increasing its length by one.
         *
         * @param labels A pair of edge labels, forward edge then reverse edge.  
         */
        void addNodeToEnd(std::pair<IOType, IOType> labels)
        {
            // Add new node (N+1, with N not yet incremented) to end of graph 
            std::stringstream ssi, ssj;
            ssi << this->N;
            ssj << this->N + 1;
            this->__addNode(ssj.str());

            // Add edges N -> N+1 and N+1 -> N (with N not yet incremented)
            this->__addEdge(ssi.str(), ssj.str(), labels.first);
            this->__addEdge(ssj.str(), ssi.str(), labels.second);

            // Separately keep track of the new edge labels  
            this->line_labels.emplace_back(
                std::make_pair(
                    static_cast<InternalType>(labels.first),
                    static_cast<InternalType>(labels.second)
                )
            );

            // Increment N 
            this->N++; 
        }

        /**
         * Remove the last node (N) from the graph (along with the two edges), 
         * decreasing its length by one.
         *
         * @throws std::runtime_error if `this->N` is zero. 
         */
        void removeNodeFromEnd()
        {
            // Throw an exception if N == 0, in which case removal is impossible
            if (this->N == 0)
                throw std::runtime_error("Line graph cannot be empty");

            // Delete N from this->order
            this->order.pop_back(); 

            // Run through and delete all edges with N as source
            std::stringstream ss; 
            ss << this->N; 
            Node* node = this->getNode(ss.str()); 
            this->edges[node].clear();

            // Run through and delete all edges with N as target 
            for (auto&& v : this->edges)
            {
                // v.first of type Node*, v.second of type std::unordered_map<Node*, InternalType>
                for (auto it = v.second.begin(); it != v.second.end(); ++it)
                {
                    if (it->first == node)
                    {
                        (v.second).erase(it);
                        break;
                    }
                } 
            }

            // Erase N from this->edges 
            for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
            {
                if (it->first == node)
                {
                    this->edges.erase(it);
                    break;
                }
            }
           
            // Delete the heap-allocated Node itself 
            delete node;
            this->nodes.erase(ss.str());
            this->numnodes--;

            // Remove the last pair of edge labels
            this->line_labels.pop_back(); 

            // Decrement N
            this->N--;  
        }

        /**
         * Ban node addition via `addNode()`: nodes can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param id ID of node to be added (to match signature with parent method). 
         * @throws std::runtime_error if invoked at all. 
         */
        Node* addNode(std::string id)
        {
            throw std::runtime_error("LabeledDigraph<...>::addNode() cannot be called; use LineGraph<...>::addNodeFromEnd()");
            return nullptr; 
        }

        /**
         * Ban node removal via `removeNode()`: nodes can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param id ID of node to be removed (to match signature with parent method). 
         * @throws std::runtime_error if invoked at all. 
         */
        void removeNode(std::string id)
        {
            throw std::runtime_error("LabeledDigraph<...>::removeNode() cannot be called; use LineGraph<...>::removeNodeFromEnd()"); 
        }

        /**
         * Ban edge addition via `addEdge()`: edges can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param source_id ID of source node of new edge (to match signature 
         *                  with parent method). 
         * @param target_id ID of target node of new edge (to match signature 
         *                  with parent method). 
         * @param label     Label on new edge (to match signature with parent method).
         * @throws std::runtime_error if invoked at all. 
         */
        void addEdge(std::string source_id, std::string target_id, IOType label = 1)
        {
            throw std::runtime_error("LabeledDigraph<...>::addEdge() cannot be called; use LineGraph<...>::addNodeToEnd()"); 
        }

        /**
         * Ban edge removal via `removeEdge()`: edges can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param source_id ID of source node of edge to be removed (to match 
         *                  signature with parent method). 
         * @param target_id ID of target node of edge to be removed (to match
         *                  signature with parent method).
         * @throws std::runtime_error if invoked at all.
         */
        void removeEdge(std::string source_id, std::string target_id)
        {
            throw std::runtime_error("LabeledDigraph<...>::removeEdge() cannot be called; use LineGraph<...>::removeNodeFromEnd()"); 
        }

        /**
         * Set the edge labels between the i-th and (i+1)-th nodes (i -> i+1
         * then i+1 -> i) to the given values.
         *
         * @param i      Index of edge labels to update. 
         * @param labels A pair of edge labels, i -> i+1 then i+1 -> i. 
         */
        void setEdgeLabels(const unsigned i, std::pair<IOType, IOType> labels)
        {
            std::stringstream ssi, ssj;
            ssi << i;
            ssj << i + 1;
            this->setEdgeLabel(ssi.str(), ssj.str(), labels.first);
            this->setEdgeLabel(ssj.str(), ssi.str(), labels.second);
            this->line_labels[i].first = static_cast<InternalType>(labels.first);
            this->line_labels[i].second = static_cast<InternalType>(labels.second); 
        }

        /**
         * Compute the probability of exiting the line graph through the upper 
         * node, `this->N` (to an auxiliary "upper exit" node), rather than 
         * through the lower node, `0` (to an auxiliary "lower exit" node), 
         * starting from `0`.  
         * 
         * @param lower_exit_rate Rate of exit through the lower node (`0`). 
         * @param upper_exit_rate Rate of exit through the upper node (`this->N`).
         * @returns               Probability of exit from `0` through `this->N`.
         */
        IOType getUpperExitProb(IOType lower_exit_rate, IOType upper_exit_rate) 
        {   
            // Start with label(N -> N-1) / label(N -> exit), then add one
            InternalType _lower_exit_rate = static_cast<InternalType>(lower_exit_rate);
            InternalType _upper_exit_rate = static_cast<InternalType>(upper_exit_rate); 
            InternalType invprob = this->line_labels[this->N-1].second / _upper_exit_rate; 
            invprob += 1;  
            for (int i = this->N - 1; i > 0; --i)
            {
                // Multiply by label(i -> i-1) / label(i -> i+1) then add one
                // for i = N-1, ..., 1
                invprob *= (this->line_labels[i-1].second / this->line_labels[i].first); 
                invprob += 1; 
            }
            // Finally multiply by label(0 -> exit) / label(0 -> 1), then add one
            invprob *= (_lower_exit_rate / this->line_labels[0].first);
            invprob += 1; 

            return static_cast<IOType>(1 / invprob);  
        }

        /**
         * Compute the reciprocal of the *unconditional* mean first-passage
         * time to exit from the line graph through the upper node, `this->N`
         * (to an auxiliary "upper exit" node), starting from `0`, given that
         * exit through the lower node, `0`, is impossible. 
         *
         * @param upper_exit_rate Rate of exit through the upper node (`this->N`).
         * @returns               Reciprocal of mean first-passage time from
         *                        `0` to exit through `this->N`. 
         */
        IOType getUpperExitRate(IOType upper_exit_rate)
        {
            // Start with label(1 -> 0) / label(0 -> 1), then add one 
            InternalType _upper_exit_rate = static_cast<InternalType>(upper_exit_rate); 
            invrate = this->line_labels[0].second / this->line_labels[0].first;  
            invrate += 1; 
            for (int i = 1; i < this->N; ++i)
            {
                // Multiply by label(i+1 -> i) / label(i -> i+1) then add one
                // for i = 1, ..., N-1
                invrate *= (this->line_labels[i].second / this->line_labels[i].first);
                invrate += 1; 
            }

            // Finally divide by label(N -> exit) and take the reciprocal
            return static_cast<IOType>(_upper_exit_rate / invrate);  
        }

        /**
         * Compute the reciprocal of the *unconditional* mean first-passage
         * time to exit from the line graph through the lower node, `0`
         * (to an auxiliary "lower exit" node), starting from `0`, given that
         * exit through the upper node, `this->N`, is impossible.
         *
         * @param lower_exit_rate Rate of exit through the lower node (`0`). 
         * @returns               Reciprocal of mean first-passage time from
         *                        `0` to exit through `0`.   
         */
        IOType getLowerExitRate(IOType lower_exit_rate)
        {
            // Start with label(N-1 -> N) / label(N -> N-1), then add one
            InternalType _lower_exit_rate = static_cast<InternalType>(lower_exit_rate); 
            InternalType invrate = this->line_labels[this->N-1].first / this->line_labels[this->N-1].second;
            invrate += 1; 
            for (int i = this->N - 2; i >= 0; --i)
            {
                // Multiply by label(i -> i+1) / label(i+1 -> i) then add one
                // for i = N-2, ..., 0
                invrate *= (this->line_labels[i].first / this->line_labels[i].second);
                invrate += 1; 
            }

            // Finally divide by label(0 -> exit) and take the reciprocal
            return static_cast<IOType>(_lower_exit_rate / invrate);  
        }

        /**
         * Compute the reciprocal of the *conditional* mean first-passage
         * time to exit from the line graph through the upper node, `this->N`
         * (to an auxiliary "upper exit" node), starting from `0`, given that
         * exit through the upper node indeed occurs. 
         *
         * @param lower_exit_rate Rate of exit through the lower node (`0`). 
         * @param upper_exit_rate Rate of exit through the upper node (`this->N`).
         * @returns               Reciprocal of conditional mean first-passage
         *                        time from `0` to exit through `this->N`.
         */
        IOType getUpperExitRate(IOType lower_exit_rate, IOType upper_exit_rate) 
        {
            // Initialize the two recurrences for the numerator
            std::vector<InternalType> recur1, recur2;
            for (unsigned i = 0; i <= this->N; ++i)
            {
                recur1.push_back(1); 
                recur2.push_back(1);
            }

            // Running product of label(N -> exit), label(N-1 -> N), label(N-2 -> N-1), ...
            InternalType _upper_exit_rate = static_cast<InternalType>(upper_exit_rate); 
            InternalType prod1 = _upper_exit_rate;

            // Running product of label(0 -> exit), label(1 -> 0), label(2 -> 1), ...
            InternalType _lower_exit_rate = static_cast<InternalType>(lower_exit_rate); 
            InternalType prod2 = _lower_exit_rate;

            // Apply first recurrence for i = N-1: multiply by label(N -> N-1), then add label(N -> exit)
            recur1[this->N-1] = this->line_labels[this->N-1].second + prod1;

            // Apply second recurrence for i = 1: multiply by label(0 -> 1), then add label(0 -> exit)
            recur2[1] = this->line_labels[0].first + prod2;  

            // Apply the first recurrence for i = N-2, ..., 0, keeping track of 
            // each term in the recurrence
            for (int i = this->N - 2; i >= 0; ++i)
            {
                // Multiply by label(i+1 -> i)
                recur1[i] = recur1[i+1] * this->line_labels[i].second;  

                // Then add the product of label(i+1 -> i+2), ..., label(N-1 -> N),
                // label(N -> exit)
                prod1 *= this->line_labels[i+1].first;     // rhs == label(i+1 -> i+2)
                recur1[i] += prod1; 
            }

            // Apply the second recurrence for i = 2, ..., N, keeping track of 
            // each term in the recurrence
            for (int i = 2; i <= this->N; ++i)
            {
                // Multiply by label(i-1 -> i)
                recur2[i] = recur2[i-1] * this->line_labels[i-1].first;

                // Then add the product of label(0 -> exit), label(1 -> 0),
                // ..., label(i-1 -> i-2)
                prod2 *= this->line_labels[i-2].second;    // rhs == label(i-1 -> i-2) 
                recur2[i] += prod2; 
            }

            // Apply the second recurrence once more to obtain the denominator: 
            // multiply by label(N -> exit) ... 
            InternalType denom = recur2[this->N] * _upper_exit_rate;

            // ... then add the product of label(0 -> exit), label(1 -> 0), ...,
            // label(N -> N-1)
            prod2 *= this->line_labels[this->N-1].second;   // rhs == label(N -> N-1)
            denom += prod2;  

            // Compute the numerator
            InternalType numer = recur1[0] + recur2[0]; 
            for (int i = 1; i <= this->N; ++i)
                numer += (recur1[i] * recur2[i]); 

            // Now give the reciprocal of the mean first-passage time (i.e., 
            // denominator / numerator) 
            return static_cast<IOType>(denom / numer);  
        }
};

#endif 