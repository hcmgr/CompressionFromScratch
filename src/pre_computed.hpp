#ifndef PRE_COMPUTED_H
#define PRE_COMPUTED_H

#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8

/**
 * Contains all pre-computed elements of JPEG compression
 */
class JpegElements {
public:
    /**
     * JPEG quantisation matrix - can be freely replaced with others
     */
    float QUANTISATION_MATRIX[BLOCK_SIZE][BLOCK_SIZE] = {
        {1,  1,  2,  4,  8,  16, 32, 64},
        {1,  1,  2,  4,  8,  16, 32, 64},
        {2,  2,  2,  4,  8,  16, 32, 64},
        {4,  4,  4,  4,  8,  16, 32, 64},
        {8,  8,  8,  8,  8,  16, 32, 64},
        {16, 16, 16, 16, 16, 16, 32, 64},
        {32, 32, 32, 32, 32, 32, 32, 64},
        {64, 64, 64, 64, 64, 64, 64, 64}

        // {16, 11, 10, 16, 24, 40, 51, 61},
        // {12, 12, 14, 19, 26, 58, 60, 55},
        // {14, 13, 16, 24, 40, 57, 69, 56},
        // {14, 17, 22, 29, 51, 87, 80, 62},
        // {18, 22, 37, 56, 68, 109, 103, 77},
        // {24, 35, 55, 64, 81, 104, 113, 92},
        // {49, 64, 78, 87, 103, 121, 120, 101},
        // {72, 92, 95, 98, 112, 100, 103, 99}

        // {8,  6,  5,  8,  10, 14, 19, 24},
        // {6,  6,  7,  10, 13, 22, 24, 20},
        // {7,  7,  8,  14, 19, 21, 26, 22},
        // {7,  9,  11, 15, 23, 35, 32, 25},
        // {9,  11, 18, 22, 29, 41, 40, 28},
        // {12, 16, 24, 26, 33, 39, 43, 32},
        // {21, 26, 32, 35, 40, 48, 47, 38},
        // {28, 32, 36, 38, 42, 38, 40, 37}
    };

    /**
     * Stores pre-computed cosines and coefficients for DCT calculations
     */
    float dct_cosines[BLOCK_SIZE][BLOCK_SIZE];
    float dct_coefs[BLOCK_SIZE][BLOCK_SIZE];

    JpegElements(); // constructor

    ~JpegElements(); // destructor

    /**
     * Stores zig-zag ordering of block indices
     */
    std::vector<std::pair<int, int>> zig_zag_indices;

    /**
     * Returns quant. matrix as a cv::Mat
     */
    cv::Mat get_quantisation_matrix();

    /**
     * Populates pre-computed DCT cosines matrix
     */
    void populate_dct_cosines_matrix();

    /**
     * Populates pre-computed DCT coefficients matrix
     */
    void populate_dct_coefs_matrix();

    /**
     * Populates pre-computed zig-zag indices
     */
    void populate_zig_zag_indices();
};

#endif // PRE_COMPUTED_H