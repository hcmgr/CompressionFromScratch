#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

#include "pre_computed.hpp"
#include "huffman.hpp"
#include "shared.hpp"

template<typename T>
void print_vector(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]" << std::endl;
}

std::string getDiv2kFileName(int number) {
    std::ostringstream filename;
    filename << "DIV2K_train_HR/" << std::setfill('0') << std::setw(4) << number << ".png";
    return filename.str();
}

/**
 * Convert image from [B,G,R] to [Y,Cb,Cr] format
 */
cv::Mat bgr_to_ycbcr(cv::Mat bgr_image) {
    cv::Mat ycbcr_image(bgr_image.rows, bgr_image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    float b, g, r;
    float y, cb, cr;
    for (int i = 0; i < bgr_image.rows; i++) {
        for (int j = 0; j < bgr_image.cols; j++) {
            cv::Vec3b &bgr_pixels = bgr_image.at<cv::Vec3b>(i, j);
            b = bgr_pixels[0]; 
            g = bgr_pixels[1]; 
            r = bgr_pixels[2];

            y = 0.299 * r + 0.587 * g + 0.114 * b;
            cb = 0.564 * (b - y);
            cr = 0.713 * (r - y);

            cv::Vec3b &ycbcr_pixels = ycbcr_image.at<cv::Vec3b>(i, j);
            ycbcr_pixels[0] = round(y); // Y
            ycbcr_pixels[1] = round(cb); // Cb
            ycbcr_pixels[2] = round(cr); // Cr
        }
    }
    return ycbcr_image;
}

/**
 * Performs DCT step on the given 8x8 block
 */
void dct_block(cv::Mat block, JpegElements &jpegElements) {
    float temp;
    int N = block.rows;
    int r,c,i,j;
    for (r = 0; r < N; r++) {
        for (c = 0; c < N; c++) {
            temp = 0.0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    temp += jpegElements.dct_cosines[r][i] * 
                            jpegElements.dct_cosines[c][j] * 
                            block.at<float>(i, j);
                }
            }
            temp *= jpegElements.dct_coefs[r][c];
            block.at<float>(r, c) = round(temp);
        }
    }
}

/**
 * Performs quantisation step on the given 8x8 block
 */
void quantise_block(cv::Mat block, JpegElements &jpegElements) {
    cv::Mat quantisationMatrix = jpegElements.get_quantisation_matrix();

    if (block.rows != quantisationMatrix.rows || block.cols != quantisationMatrix.cols) {
        std::cout << "Fatal: incorrect block size" << std::endl;
        return;
    }

    for (int r = 0; r < BLOCK_SIZE; r++) {
        for (int c = 0; c < BLOCK_SIZE; c++) {
            float currIntensity = block.at<float>(r, c);
            block.at<float>(r, c) = round(currIntensity / quantisationMatrix.at<float>(r, c));
        }
    }
}

/**
 * Convert 8x8 block into zig-zag ordered 64-d array
 */
std::vector<int> block_to_zig_zag(cv::Mat block, JpegElements &jpegElements) {
    std::vector<int> res;

    if (jpegElements.zig_zag_indices.size() != BLOCK_SIZE*BLOCK_SIZE) {
        std::cout << "Fatal: incorrect block size" << std::endl;
        return res;
    }
    
    int r, c;
    int val = 0;
    std::pair<int, int> pos;

    for (int i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++) {
        pos = jpegElements.zig_zag_indices[i];
        r = pos.first; c = pos.second;
        val = static_cast<int>(block.at<float>(r, c));
        res.push_back(val);
    }

    return res;
}

/**
 * Run-length-encoding of given block array
 */
std::vector<int> rle(std::vector<int> block_array) {
    std::vector<int> rle_array;

    int n = block_array.size();
    int i = 0, j = 0;
    while (i < n) {
        j = i+1;
        while (block_array[j] == block_array[i]) {
            j++;
        }
        rle_array.push_back((j - i)); // count
        rle_array.push_back((block_array[i])); // number
        i = j;
    }

    return rle_array;
}

/**
 * Apply jpeg steps for given block:
 *  - normalise
 *  - DCT
 *  - quantise
 *  - zig-zag
 *  - entropy encode
 */
int jpeg_block(cv::Mat block, JpegElements &jpegElements) {
    std::cout << "Init" << std::endl << block << std::endl << std::endl;

    // pre-process
    block.convertTo(block, CV_32F);
    block -= 128;
    std::cout << "Normalised" << std::endl << block << std::endl << std::endl;

    // dct
    dct_block(block, jpegElements);
    std::cout << "DCT'd" << std::endl << block << std::endl << std::endl;

    // quantise block
    quantise_block(block, jpegElements);
    std::cout << "Quantised" << std::endl << block << std::endl << std::endl;

    // convert to flattened uchar array in zig-zag order
    std::vector<int> block_array = block_to_zig_zag(block, jpegElements);
    std::cout << "Zig-zag encoded" << std::endl;
    print_vector(block_array);

    // rle encode
    std::vector<int> rle_block_array = rle(block_array);
    std::cout << "RLE encoded" << std::endl;
    print_vector(rle_block_array);
    std::cout << "Size: " << rle_block_array.size() << std::endl;
    return 0;
}



int jpeg() {
    JpegElements jpegElements = JpegElements();

    // load image
    // int num = 28;
    // std::string filename = getDiv2kFileName(num);
    std::string filename = "images/test_1.jpg";
    cv::Mat image = CvImageUtils::loadImage(filename);
    // display_image(image, "");

    // convert to Y, Cr, Cb format
    cv::Mat new_image = bgr_to_ycbcr(image);
    // display_image(new_image, "");

    // split into channels
    std::vector<cv::Mat> channels;
    split(image, channels);
    cv::Mat curr_channel = channels[0]; // Y
    
    // extract block and apply jpeg on it
    int r = image.rows / 3, c = image.cols / 3;
    cv::Rect block_rect(r, c, 8, 8);
    cv::Mat block = curr_channel(block_rect).clone();
    jpeg_block(block, jpegElements);

    // TODO: concat block arrays to produce channel encodings
    // TODO: concat channel encodings to produce final data
    return 0;
}

void test_huffman_tree_build() {
    Huffman h;
    std::vector<int> rle_data = {4,1,3,2,2,3,1,4}; // 1,1,1,1,2,2,2,3,3,4
    std::vector<uchar> enc_data = h.encode_data(rle_data);
}

int main() {
    test_huffman_tree_build();
    return 0;
}