#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <map>
#include <queue>
#include <opencv2/opencv.hpp>

/**
 * Class to rep. a node in a huffman encoding tree
 */
class HuffmanNode {
public:
    int val;
    int freq;
    HuffmanNode *left;
    HuffmanNode *right;

    // leaf constructor
    HuffmanNode(int val, int freq);

    // non-leaf constructor
    HuffmanNode(HuffmanNode *left, HuffmanNode *right);
};

/**
 * Class to perform huffman coding on a given byte array.
 */
class Huffman {
private:
    /* Huffman encoding tree */
    HuffmanNode *root; 

    /* Encodings map (i.e. mapping of the form: byte -> binary string) */
    std::map<int, std::string> encodings;

    /**
     * Builds the huffman encoding tree
     */
    void build_huffman_tree(std::vector<int> data);

    /**
     * Traverses the huffman encoding tree to populate the encodings map
     */
    void build_encodings_map(HuffmanNode *root, std::string code);

public:
    /**
     * Huffman encodes the given byte array
     */
    std::vector<uchar> encode_data(std::vector<int> data, bool debug);

    std::map<int, std::string> get_encodings();
};

#endif // HUFFMAN_H