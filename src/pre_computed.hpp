#pragma once

#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 8
#define NUM_QUANT_MATRICES 5

//
// Holds pre-computed elements of JPEG compression, namely:
//      - quantisation matrix
//      - dct cosines
//      - dct coefficients
//      - zig-zag ordering of indices
//
class JpegElements {
public:
    //
    // JPEG quantisation matrices.
    // Listed from least aggresive to most aggressive.
    //
    float QUANTISATION_MATRIX[NUM_QUANT_MATRICES][BLOCK_SIZE][BLOCK_SIZE] = {
        // no quantisation
        { 
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1}
        },

        {   
            {1,  2,  3,  4,  6,  8,  12,  16},
            {2,  2,  3,  4,  6,  8,  12,  16},
            {3,  3,  4,  4,  6,  8,  12,  16},
            {4,  4,  4,  5,  6,  8,  12,  16},
            {6,  6,  6,  6,  6,  8,  12,  16},
            {8,  8,  8,  8,  8,  8,  12,  16},
            {12, 12, 12, 12, 12, 12, 12,  16},
            {16, 16, 16, 16, 16, 16, 16,  16}
        },

        {   
            {1,  2,  3,  5,  8,  12,  20,  32},
            {2,  2,  3,  5,  8,  12,  20,  32},
            {3,  3,  4,  5,  8,  12,  20,  32},
            {5,  5,  5,  6,  8,  12,  20,  32},
            {8,  8,  8,  8,  8,  12,  20,  32},
            {12, 12, 12, 12, 12, 12,  20,  32},
            {20, 20, 20, 20, 20, 20,  20,  32},
            {32, 32, 32, 32, 32, 32,  32,  32}
        },

        // best performing one by far
        {
            {1,  1,  2,  4,  8,  16, 32, 64},
            {1,  1,  2,  4,  8,  16, 32, 64},
            {2,  2,  2,  4,  8,  16, 32, 64},
            {4,  4,  4,  4,  8,  16, 32, 64},
            {8,  8,  8,  8,  8,  16, 32, 64},
            {16, 16, 16, 16, 16, 16, 32, 64},
            {32, 32, 32, 32, 32, 32, 32, 64},
            {64, 64, 64, 64, 64, 64, 64, 64}
        },

        {
            {16, 11, 10, 16, 24, 40, 51, 61},
            {12, 12, 14, 19, 26, 58, 60, 55},
            {14, 13, 16, 24, 40, 57, 69, 56},
            {14, 17, 22, 29, 51, 87, 80, 62},
            {18, 22, 37, 56, 68, 109, 103, 77},
            {24, 35, 55, 64, 81, 104, 113, 92},
            {49, 64, 78, 87, 103, 121, 120, 101},
            {72, 92, 95, 98, 112, 100, 103, 99}
        }
    };

    //
    // Pre-computed cosines and coefficients of DCT calculations
    //
    float dct_cosines[BLOCK_SIZE][BLOCK_SIZE];
    float dct_coefs[BLOCK_SIZE][BLOCK_SIZE];

    JpegElements();
    ~JpegElements() = default;

    //
    // Zip-zag ordering of block indices
    //
    std::vector<std::pair<int, int>> zig_zag_indices;

    //
    // Returns the ith quantisation matrix (see pre_computed.hpp)
    // as a cv::Mat object
    //
    cv::Mat getQuantisationMatrix(int i);

    //
    // Populates the pre-computed DCT cosines matrix
    //
    void populateDctCosinesMatrix();

    //
    // Populates the pre-computed DCT coefficients matrix
    //
    void populateDctCoefsMatrix();

    //
    // Populates the pre-computed zig-zag indices
    //
    void populateZigZagIndices();
};