#include <map>
#include <queue>
#include <opencv2/opencv.hpp>
#include "huffman.hpp"
#include "shared.hpp"
#include "rle.hpp"

HuffmanNode::HuffmanNode(int val, int freq) { // leaf
    this->val = val;
    this->freq = freq;
    this->left = nullptr;
    this->right = nullptr;
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
 * Builds the huffman encoding tree
 * 
 * NOTE: expects data to be run-length-encoded
 */
void Huffman::build_huffman_tree(std::vector<int> rle_data) {
    // build up frequency map
    std::map<int, int> freqCount;
    int n = rle_data.size();
    int f, el;
    for (int i = 0; i < n; i+=2) {
        f = rle_data[i], el = rle_data[i+1];
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
    this->root = pq.top().second;
}

/**
 * Traverses huffman tree to find encodings for each leaf
 */
void Huffman::build_encodings_map(HuffmanNode *root, std::string code) {
    if (!root) {
        return;
    }

    if (!root->left && !root->right) {
        this->encodings[root->val] = code;
    }

    build_encodings_map(root->left, code + "0");
    build_encodings_map(root->right, code + "1");
}

/**
 * Huffman encodes the given byte array
 */
std::vector<uchar> Huffman::encode_data(std::vector<int> data, bool debug) {
    // rle encode the data
    std::vector<int> rle_data = Rle::rle_encode(data);

    // calculate encodings
    build_huffman_tree(rle_data);
    build_encodings_map(this->root, "");

    // convert byte array to its encoded binary string
    std::string binary_string;
    for (int byte : data) {
        binary_string += this->encodings[byte];
    }

    // convert new binary string into byte array
    std::vector<uchar> byte_array;
    std::string byte_string;
    int val;
    for (size_t i = 0; i < binary_string.size(); i += 8) {
        byte_string = binary_string.substr(i, 8);
        val = std::stoi(byte_string, nullptr, 2);
        byte_array.push_back(val);
    }

    if (debug) {
        std::cout << std::endl << "Data" << std::endl;
        PrintUtils::print_vector(data);
        std::cout << std::endl << "Encodings map" << std::endl;
        PrintUtils::print_map(encodings);
        std::cout << std::endl << "Full binary string: (" << binary_string.size() << ")" << std::endl;
        std::cout << binary_string << std::endl;
    }

    return byte_array;
}

std::map<int, std::string> Huffman::get_encodings() {
    return this->encodings;
}

///// TESTING /////

