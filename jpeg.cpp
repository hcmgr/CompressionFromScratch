#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

#include "pre_computed.hpp"
#include "huffman.hpp"
#include "shared.hpp"
#include "rle.hpp"

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
    int u,v,i,j;
    for (u = 0; u < N; u++) {
        for (v = 0; v < N; v++) {
            temp = 0.0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    // temp += jpegElements.dct_cosines[u][i] * 
                    //         jpegElements.dct_cosines[v][j] * 
                    //         block.at<float>(i, j);
                    temp += block.at<float>(i, j);
                }
            }
            temp *= jpegElements.dct_coefs[u][v];
            block.at<float>(u, v) = round(temp);
        }
    }
}

/**
 * Performs the inverse DCT step on the given 8x8 block
 */
void inverse_dct_block(cv::Mat block, JpegElements &jpegElements) {
    float temp;
    int N = block.rows;
    int i,j,u,v;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0.0;
            for (u = 0; u < N; u++) {
                for (v = 0; v < N; v++) {
                    temp += jpegElements.dct_cosines[i][u] * 
                            jpegElements.dct_cosines[j][v] * 
                            block.at<float>(u, v);
                }
            }
            temp *= jpegElements.dct_coefs[u][v];
            block.at<float>(u, v) = round(temp);
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
    PrintUtils::print_vector(block_array);
    
    // huffman encode
    Huffman h;
    std::vector<uchar> huff_encoded_data = h.encode_data(block_array);
    std::cout << std::endl << "Final byte array: (" << huff_encoded_data.size() << ")" << std::endl;
    PrintUtils::print_vector(huff_encoded_data);

    return 0;
}

int jpeg() {
    JpegElements jpegElements = JpegElements();
    std::string filename = "images/test_1.jpg";
    cv::Mat image = CvImageUtils::loadImage(filename);
    CvImageUtils::display_image(image, "");

    // convert to Y, Cr, Cb format
    cv::Mat new_image = bgr_to_ycbcr(image);
    CvImageUtils::display_image(new_image, "");

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
    std::vector<int> data = {1,1,2,2,3,3,4,4};
    std::vector<uchar> enc_data = h.encode_data(data);
}

void test_dct_inverse() {
    JpegElements jpegElements = JpegElements();

    // initialise random 8x8 block
    cv::Mat rand_block(8, 8, CV_8UC1);
    cv::randu(rand_block, 0, 256);
    std::cout << rand_block << std::endl << std::endl;

    // dct
    dct_block(rand_block, jpegElements);
    // std::cout << rand_block << std::endl << std::endl;

    // inverse dct
    // inverse_dct_block(rand_block, jpegElements);
    // std::cout << rand_block << std::endl;
}

int main() {
    // jpeg();
    // test_huffman_tree_build();
    test_dct_inverse();
    return 0;
}