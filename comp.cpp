#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

void print_image_stats(cv::Mat& image) {
    std::cout << "NROWS: " << image.rows << std::endl;
    std::cout << "NCOLS: " << image.cols << std::endl;
    std::cout << "NCHANNELS: " << image.channels() << std::endl;
}

void display_image(cv::Mat& image, std::string name) {
    print_image_stats(image);
    cv::imshow(name, image);
    cv::waitKey(0);
}

#define BLOCK_SIZE 8

// /**
//  * A JPEG quantisation matrix - can be freely replaced with others
//  */
// float QUANTISATION_MATRIX[BLOCK_SIZE][BLOCK_SIZE] = {
//     {16, 11, 10, 16, 24, 40, 51, 61},
//     {12, 12, 14, 19, 26, 58, 60, 55},
//     {14, 13, 16, 24, 40, 57, 69, 56},
//     {14, 17, 22, 29, 51, 87, 80, 62},
//     {18, 22, 37, 56, 68, 109, 103, 77},
//     {24, 35, 55, 64, 81, 104, 113, 92},
//     {49, 64, 78, 87, 103, 121, 120, 101},
//     {72, 92, 95, 98, 112, 100, 103, 99}
// };

// /**
//  * A JPEG quantisation matrix - can be freely replaced with others
//  */
// float QUANTISATION_MATRIX[BLOCK_SIZE][BLOCK_SIZE] = {
//     {8,  6,  5,  8,  10, 14, 19, 24},
//     {6,  6,  7,  10, 13, 22, 24, 20},
//     {7,  7,  8,  14, 19, 21, 26, 22},
//     {7,  9,  11, 15, 23, 35, 32, 25},
//     {9,  11, 18, 22, 29, 41, 40, 28},
//     {12, 16, 24, 26, 33, 39, 43, 32},
//     {21, 26, 32, 35, 40, 48, 47, 38},
//     {28, 32, 36, 38, 42, 38, 40, 37}
// };

/**
 * A JPEG quantisation matrix - can be freely replaced with others
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
 * Pre-computed cosines for DCT
 */
float DCT_COSINES[BLOCK_SIZE][BLOCK_SIZE];

/**
 * Pre-computed coefficients for DCT
 */
float DCT_COEFS[BLOCK_SIZE][BLOCK_SIZE];

/**
 * Populate pre-computed DCT cosines matrix
 */
void populate_dct_cosines_matrix(float cosines[BLOCK_SIZE][BLOCK_SIZE]) {
    double temp;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp = (2*i+1)*j*M_PI / 2 / BLOCK_SIZE;
            cosines[i][j] = cos(temp);
        }
    }
}

/**
 * Populate pre-computed DCT coefficients matrix
 */
void populate_dct_coefs_matrix(float coefs[BLOCK_SIZE][BLOCK_SIZE]) {
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
            coefs[i][j] = temp;
        }
    }
}

/**
 * Convert image from [B,G,R] to [Y,Cb,Cr] format
 */
cv::Mat bgr_to_ycbcr(cv::Mat bgr_image) {
    cv::Mat ycbcr_image(bgr_image.rows, bgr_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    float b, g, r;
    for (int i = 0; i < bgr_image.rows; i++) {
        for (int j = 0; j < bgr_image.cols; j++) {
            cv::Vec3b &bgr_pixels = bgr_image.at<cv::Vec3b>(i, j);
            b = bgr_pixels[0]; 
            g = bgr_pixels[1]; 
            r = bgr_pixels[2];

            cv::Vec3b &ycbcr_pixels = ycbcr_image.at<cv::Vec3b>(i, j);
            ycbcr_pixels[0] = 0.299*r + 0.587*g + 0.114*b; // Y
            ycbcr_pixels[1] = 0.564*(b - ycbcr_pixels[0]); // Cb
            ycbcr_pixels[2] = 0.713*(r - ycbcr_pixels[0]); // Cr
        }
    }
    return ycbcr_image;
}

/**
 * Performs DCT step on the given 8x8 block
 */
void dct_block(cv::Mat block) {
    float temp;
    int N = block.rows;
    int r,c,i,j;
    for (r = 0; r < N; r++) {
        for (c = 0; c < N; c++) {
            temp = 0.0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    temp += (DCT_COSINES[r][i] * DCT_COSINES[c][j] * block.at<float>(i, j));
                }
            }
            temp *= DCT_COEFS[r][c];
            block.at<float>(r, c) = round(temp);
        }
    }
}

/**
 * Performs quantisation step on the given 8x8 block
 */
cv::Mat quantise_block(cv::Mat block, cv::Mat quantisationMatrix) {
    if (block.rows != quantisationMatrix.rows || block.cols != quantisationMatrix.cols) {
        std::cout << "Fatal: incorrect block size" << std::endl;
    }

    for (int r = 0; r < BLOCK_SIZE; r++) {
        for (int c = 0; c < BLOCK_SIZE; c++) {
            float currIntensity = block.at<float>(r, c);
            block.at<float>(r, c) = round(currIntensity / quantisationMatrix.at<float>(r, c));
        }
    }

    return block;
}

/**
 * Compute and return indices of 8x8 zig zag traversal
 * 
 * NOTE: think this works in general, have hardcoded for now
 */
std::vector<std::pair<int,int>> zig_zag_indices() {
    std::vector<std::pair<int, int>> indices;
    indices.push_back(std::make_pair(0, 0));

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

        indices.push_back(std::make_pair(r, c));
        cnt++;
    }

    return indices;
}

/**
 * Run-length-encoding of given block array
 */
std::vector<uchar> rle(std::vector<uchar> block_array) {
    std::vector<uchar> rle_array;

    int n = block_array.size();
    int i = 0, j = 0;
    while (i < n) {
        j = i+1;
        while (block_array[j] == block_array[i]) {
            j++;
        }
        rle_array.push_back((j - 1)); // count
        rle_array.push_back((block_array[i])); // number
        i = j;
    }

    return rle_array;
}

/**
 * Perform huffman encoding on the given rle-encoded array of bytes
 */
std::vector<uchar> huffman(std::vector<uchar> rle_array);

cv::Mat loadImageFromDiv2k(int number) {
    std::ostringstream filename;

    filename << "DIV2K_train_HR/" << std::setfill('0') << std::setw(4) << number << ".png";
    cv::Mat image = cv::imread(filename.str(), cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not open or find the image: " << filename.str() << std::endl;
    }
    return image;
}

int muckin() {
    int num = 28;
    cv::Mat image = loadImageFromDiv2k(num);
    display_image(image, std::to_string(num));

    // split into channels
    // std::vector<cv::Mat> channels;
    // split(image, channels);
    // cv::Mat blue_channel = channels[2];

    // extracting 8x8 block
    // cv::Rect bgr_roi_rect(image.rows/2, image.cols/2, 8, 8);
    // cv::Mat bgr_block = blue_channel(bgr_roi_rect).clone();
    // std::cout << bgr_block << std::endl;

    // use Y, Cb and Cr instead
    cv::Mat ycbcr_image = bgr_to_ycbcr(image);
    display_image(ycbcr_image, std::to_string(num));
    // cv::Rect ycbcr_roi_rect(image.rows/2, image.cols/2, 8, 8);
    // cv::Mat ycbcr_block = blue_channel(ycbcr_roi_rect).clone();
    // std::cout << ycbcr_block << std::endl;

    // // float format needed for high precision in dft and quantisation steps
    // cv::Mat float_block;
    // block.convertTo(float_block, CV_32F);
    // std::cout << float_block << std::endl;
    
    // // subtract 128 
    // cv::subtract(float_block, cv::Scalar(128), float_block);
    // std::cout << float_block << std::endl;

    // // DCT
    // populate_dct_cosines_matrix(DCT_COSINES);
    // populate_dct_coefs_matrix(DCT_COEFS);
    // dct_block(float_block);
    // std::cout << float_block << std::endl;

    // // quantise block
    // cv::Mat quantisation_matrix(BLOCK_SIZE, BLOCK_SIZE, CV_32F, QUANTISATION_MATRIX);
    // cv::Mat quantised_block = quantise_block(float_block, quantisation_matrix);
    // std::cout << quantised_block << std::endl;

    // // flatten and convert to vector<uchar>
    // cv::Mat flattenedBlock = quantised_block.reshape(0, 1).clone();
    // uchar* data = flattenedBlock.ptr();
    // std::vector<uchar> flattenedVec(data, data + flattenedBlock.cols);

    // // rle encode vector
    // std::vector<uchar> rleVec = rle(flattenedVec);
    // for (uchar el : rleVec) {
    //     std::cout << static_cast<int>(el) << std::endl;
    // }
    // std::cout << std::endl << rleVec.size() << std::endl;

    // print zig zag indices
    // std::vector<std::pair<int, int>> inds = zig_zag_indices();
    // for (std::pair<int, int> rc : inds) {
    //     std::cout << rc.first << ", " << rc.second << std::endl;
    // }

    return 0;
}

int jpeg() {
    return 0;
}

int main() {
    muckin();
    return 0;
}


