#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8

/**
 * Contains all pre-computed elements of JPEG compression
 */
class JPEGPreComputedElements {
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
    };

    /**
     * Pre-computed cosines and coefficients for DCT calculation
     */
    float dct_cosines[BLOCK_SIZE][BLOCK_SIZE];
    float dct_coefs[BLOCK_SIZE][BLOCK_SIZE];

    /**
     * Zig-zag ordering of block indices
     */
    std::vector<std::pair<int, int>> zig_zag_indices;

    JPEGPreComputedElements() {
        populate_dct_coefs_matrix();
        populate_dct_cosines_matrix();
        populate_zig_zag_indices();
    }

    /**
     * Returns quant. matrix as a cv::Mat
     */
    cv::Mat load_quantisation_matrix() {
        return cv::Mat(BLOCK_SIZE, BLOCK_SIZE, CV_32F, QUANTISATION_MATRIX);
    }

    /**
     * Populate pre-computed DCT cosines matrix
     */
    void populate_dct_cosines_matrix() {
        double temp;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                temp = (2*i+1)*j*M_PI / 2 / BLOCK_SIZE;
                dct_cosines[i][j] = cos(temp);
            }
        }
    }

    /**
     * Populate pre-computed DCT coefficients matrix
     */
    void populate_dct_coefs_matrix() {
        float temp;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                temp = 1 / sqrt(2 * BLOCK_SIZE);
                if (i == 0) {
                    temp *= (1 / sqrt(2));
                }
                if (j == 0) {
                    temp *= (1 / sqrt(2));
                }
                dct_coefs[i][j] = temp;
            }
        }
    }

    /**
     * Populate pre-computed zig-zag indices
     */
    void populate_zig_zag_indices() {
        zig_zag_indices.push_back(std::make_pair(0, 0));

        int m = 8, n = 8;
        int r = 0, c = 0;
        int cnt = 1;
        int up = 1; // 1 if moving up-right, 0 if moving down-left

        while (cnt < m*n) {
            // top side
            if (r == 0) {
                if (up) {
                    c += 1; // across 1
                    up = 0;
                } else {
                    c -= 1; r += 1; // down-left 1
                }
            }

            // bottom side
            else if (r == 7) {
                if (up) {
                    c += 1; r -= 1; // up-right 1
                } else {
                    c += 1; // across 1
                    up = 1;
                }
            }

            // left side
            else if (c == 0) {
                if (up) {
                    c += 1; r -= 1; // up-right 1
                } else {
                    r += 1; // down 1
                    up = 1;
                }
            }

            // right side
            else if (c == 7) {
                if (up) {
                    r += 1; // down 1
                    up = 0;
                } else {
                    c -= 1; r += 1; // down-left 1
                }
                
            }

            // other
            else {
                if (up) {
                    c += 1; r -= 1; // up-right 1
                } else {
                    c -= 1; r += 1; // down-left 1
                }
            }

            zig_zag_indices.push_back(std::make_pair(r, c));
            cnt++;
        }
    }
};