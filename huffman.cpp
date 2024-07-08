#include <map>
#include <queue>
#include <opencv2/opencv.hpp>
#include "huffman.hpp"

HuffmanNode::HuffmanNode(uchar val, int freq) { // leaf
    this->val = val;
    this->freq = freq;
}

HuffmanNode::HuffmanNode(HuffmanNode *left, HuffmanNode *right) { // non-leaf
    this->val = 0;
    this->freq = 0;
    if (left) {
        this->freq += left->freq;
    }
    if (right) {
        this->freq += right->freq;
    }
    this->left = left;
    this->right = right;
}

/**
 * Perform huffman encoding on the given rle-encoded array
 */
HuffmanNode* build_huffman_tree(std::vector<uchar> arr) {
    // build up frequency map
    std::map<uchar, int> freqCount;
    int n = arr.size();
    int f, el;
    for (int i = 0; i < n; i+=2) {
        f = arr[i], el = arr[i+1];
        freqCount[el] += f;
    }

    // build up min heap
    std::priority_queue<std::pair<int, HuffmanNode*>> pq;
    for (auto p : freqCount) {
        el = p.first, f = p.second;
        HuffmanNode *node = new HuffmanNode(el, f);
        pq.push(std::make_pair(f*-1, node));
    }

    // build up huffman tree from min heap
    std::pair<int, HuffmanNode*> e1, e2;
    HuffmanNode *left, *right;
    while (pq.size() > 1) {
        e1 = pq.top(); pq.pop(); 
        e2 = pq.top(); pq.pop();
        left = e1.second;
        right = e2.second;
        HuffmanNode *node = new HuffmanNode(left, right);
        pq.push(std::make_pair(node->freq, node));
    }
    HuffmanNode *root = pq.top().second;
    return root;
}

void encode_leaves(HuffmanNode *root, std::string code, std::map<uchar, std::string> &encodings) {
    if (!root) {
        return;
    }

    if (!root->left && !root->right) {
        encodings[root->val] = code;
    }

    encode_leaves(root->left, code + "0", encodings);
    encode_leaves(root->right, code + "1", encodings);
}

void test_huffman_tree_build() {
    std::vector<uchar> vec = {4,1,3,2,2,3,1,4};

    HuffmanNode *root = build_huffman_tree(vec);
    std::map<uchar, std::string> encodings;
    encode_leaves(root, "", encodings);
    for (auto el : encodings) {
        std::cout << static_cast<int>(el.first) << ": " << el.second << std::endl;
    }
}