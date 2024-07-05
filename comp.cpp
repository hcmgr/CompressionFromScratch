#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>

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
float QUANTISATION_MATRIX[BLOCK_SIZE][BLOCK_SIZE] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

/**
 * Performs DCT step on the given 8x8 block
 */
cv::Mat dct_block(cv::Mat block);

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
    int num = 701;
    cv::Mat image = loadImageFromDiv2k(num);
    // display_image(image, std::to_string(num));

    // split into channels
    std::vector<cv::Mat> channels;
    split(image, channels);
    cv::Mat blue_channel = channels[0];

    // extracting 8x8 block
    cv::Rect roi_rect(image.rows/4, image.cols/4, 8, 8);
    cv::Mat block = blue_channel(roi_rect).clone();
    std::cout << block << std::endl;

    // float format needed for high precision in dft and quantisation steps
    cv::Mat float_block;
    block.convertTo(float_block, CV_32F);
    
    // subtract 128 
    cv::subtract(float_block, cv::Scalar(128), float_block);
    std::cout << float_block << std::endl;

    // quantise block
    cv::Mat quantisation_matrix(BLOCK_SIZE, BLOCK_SIZE, CV_32F, QUANTISATION_MATRIX);
    cv::Mat quantised_block = quantise_block(float_block, quantisation_matrix);
    std::cout << quantised_block << std::endl;

    // flatten and convert to vector<uchar>
    // cv::Mat flattenedBlock = block.reshape(0, 1).clone();
    // uchar* data = flattenedBlock.ptr();
    // std::vector<uchar> flattenedVec(data, data + flattenedBlock.cols);

    // rle encode vector
    // std::vector<uchar> rleVec = rle(flattenedVec);
    // for (uchar el : rleVec) {
    //     std::cout << static_cast<int>(el) << std::endl;
    // }

    // print zig zag indices
    // std::vector<std::pair<int, int>> inds = zig_zag_indices();
    // for (std::pair<int, int> rc : inds) {
    //     std::cout << rc.first << ", " << rc.second << std::endl;
    // }

    return 0;
}

int main() {
    muckin();
    return 0;
}


