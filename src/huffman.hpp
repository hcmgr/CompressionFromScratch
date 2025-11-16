#pragma once
#include <map>
#include <queue>
#include <opencv2/opencv.hpp>

//
// Huffman encoding tree node
//
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

class HuffmanEncoder {
private:
    // encoding tree
    HuffmanNode *root; 

    // encodings map, i.e. mapping of the form: byte -> binary string
    std::map<int, std::string> encodings;

    //
    // Builds up encoding tree from given byte array. Expects `data` to be rle-encoding,
    // see rle.hpp.
    //
    void buildEncodingTree(std::vector<int> data);

    //
    // Traverses huffman encoding tree to populate the encodings map
    //
    void buildEncodingsMap(HuffmanNode *root, std::string code);

public:
    //
    // Huffman encodes given byte array
    //
    std::vector<uchar> encode(std::vector<int> data, bool debug);

    std::map<int, std::string> getEncodings();
};