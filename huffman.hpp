#include <map>
#include <queue>
#include <opencv2/opencv.hpp>

/**
 * Class to rep. a node in a huffman encoding tree
 */
class HuffmanNode {
public:
    uchar val;
    int freq;
    HuffmanNode *left;
    HuffmanNode *right;

    // leaf constructor
    HuffmanNode(uchar val, int freq);

    // non-leaf constructor
    HuffmanNode(HuffmanNode *left, HuffmanNode *right);

    /**
     * 
     */
};

/**
 * Build up and return huffman encoding tree of given array
 */
HuffmanNode* build_huffman_tree(std::vector<uchar> arr);

void test_huffman_tree_build();