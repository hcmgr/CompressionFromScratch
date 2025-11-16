#include "pre_computed.hpp"

JpegElements::JpegElements() {
    populateDctCoefsMatrix();
    populateDctCosinesMatrix();
    populateZigZagIndices();
}

//
// Returns the ith quantisation matrix (see pre_computed.hpp)
// as a cv::Mat object
//
cv::Mat JpegElements::getQuantisationMatrix(int i) {
    return cv::Mat(BLOCK_SIZE, BLOCK_SIZE, CV_32F, QUANTISATION_MATRIX[i]);
}

//
// Populates the pre-computed DCT cosines matrix
//
void JpegElements::populateDctCosinesMatrix() {
    double temp;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp = (2*i+1)*j*M_PI / 16;
            dct_cosines[i][j] = cos(temp);
        }
    }
}

//
// Populates the pre-computed DCT coefficients matrix
//
void JpegElements::populateDctCoefsMatrix() {
    float temp;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp = 1;
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

//
// Populates the pre-computed zig-zag indices
//
void JpegElements::populateZigZagIndices() {
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