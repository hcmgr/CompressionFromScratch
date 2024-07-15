#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

#include "pre_computed.hpp"
#include "huffman.hpp"
#include "shared.hpp"
#include "rle.hpp"
#include "experiments.hpp"

std::string getDiv2kFileName(int number) {
    std::ostringstream filename;
    filename << "DIV2K_train_HR/" << std::setfill('0') << std::setw(4) << number << ".png";
    return filename.str();
}

/**
 * Pads given image to ensure its dimensions are a multiple of 'blockSize'
 */
cv::Mat pad_for_jpeg(cv::Mat image, int blockSize) {
    int padRows = blockSize - (image.rows % blockSize);
    int padCols = blockSize - (image.cols % blockSize);
    if (padRows == blockSize) padRows = 0;
    if (padCols == blockSize) padCols = 0;

    cv::Mat paddedImage;
    cv::copyMakeBorder(image, paddedImage, 0, padRows, 0, padCols, cv::BORDER_REPLICATE);
    return paddedImage;
}

/**
 * Convert image from [B,G,R] to [Y,Cb,Cr] format
 */
cv::Mat bgr_to_ycbcr(cv::Mat bgrImage) {
    cv::Mat ycbcrImage(bgrImage.rows, bgrImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    float b, g, r;
    float y, cb, cr;
    for (int i = 0; i < bgrImage.rows; i++) {
        for (int j = 0; j < bgrImage.cols; j++) {
            cv::Vec3b &bgrPixels = bgrImage.at<cv::Vec3b>(i, j);
            b = bgrPixels[0]; 
            g = bgrPixels[1]; 
            r = bgrPixels[2];

            y = 0.299 * r + 0.587 * g + 0.114 * b;
            cb = 128 + 0.5*b - 0.1687*r - 0.3314*g;
            cr = 128 + 0.5*r - 0.4625*g - 0.0813*b;

            cv::Vec3b &ycbcrPixels = ycbcrImage.at<cv::Vec3b>(i, j);
            ycbcrPixels[0] = MathUtils::clamp(round(y), 0, 255); // Y
            ycbcrPixels[1] = MathUtils::clamp(round(cr), 0, 255); // Cr
            ycbcrPixels[2] = MathUtils::clamp(round(cb), 0, 255); // Cb
        }
    }
    return ycbcrImage;
}

cv::Mat ycbcr_to_bgr(cv::Mat ycbcrImage) {
    cv::Mat bgrImage(ycbcrImage.rows, ycbcrImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    float y, cb, cr;
    float b, g, r;
    for (int i = 0; i < ycbcrImage.rows; i++) {
        for (int j = 0; j < ycbcrImage.cols; j++) {
            cv::Vec3b &ycbcrPixels = ycbcrImage.at<cv::Vec3b>(i, j);
            y = ycbcrPixels[0]; 
            cr = ycbcrPixels[1]; 
            cb = ycbcrPixels[2];

            cr -= 128;
            cb -= 128;

            r = y + 1.402*cr;
            g = y - 0.344*cb - 0.714*cr;
            b = y + 1.772*cb;

            cv::Vec3b &bgrPixels = bgrImage.at<cv::Vec3b>(i, j);
            bgrPixels[0] = MathUtils::clamp(round(b), 0, 255); // b
            bgrPixels[1] = MathUtils::clamp(round(g), 0, 255); // g
            bgrPixels[2] = MathUtils::clamp(round(r), 0, 255); // r
        }
    }
    return bgrImage;
}

/**
 * Performs DCT step on the given 8x8 block
 */
void dct_block(cv::Mat dctBlock, cv::Mat block, JpegElements &jpegElements) {
    float temp;
    int N = block.rows;
    int u,v,i,j;
    for (u = 0; u < N; u++) {
        for (v = 0; v < N; v++) {
            temp = 0.0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    temp += jpegElements.dct_cosines[i][u] * 
                            jpegElements.dct_cosines[j][v] * 
                            block.at<float>(i, j);
                }
            }
            temp *= (0.25) * jpegElements.dct_coefs[u][v];
            dctBlock.at<float>(u, v) = temp;
        }
    }
}

/**
 * Performs the inverse DCT step on the given 8x8 block
 */
void inverse_dct_block(cv::Mat invBlock, cv::Mat dctBlock, JpegElements &jpegElements) {
    float temp;
    int N = dctBlock.rows;
    int i,j,u,v;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0.0;
            for (u = 0; u < N; u++) {
                for (v = 0; v < N; v++) {
                    temp += jpegElements.dct_coefs[u][v] *
                            jpegElements.dct_cosines[i][u] * 
                            jpegElements.dct_cosines[j][v] * 
                            dctBlock.at<float>(u, v);
                            
                }
            }
            temp *= (0.25);
            invBlock.at<float>(i, j) = temp;
        }
    }
}

/**
 * Performs quantisation step on the given 8x8 block
 */
void quantise_block(cv::Mat quantBlock, cv::Mat dctBlock, int i, JpegElements &jpegElements) {
    cv::Mat quantisationMatrix = jpegElements.get_quantisation_matrix(i);
    for (int r = 0; r < BLOCK_SIZE; r++) {
        for (int c = 0; c < BLOCK_SIZE; c++) {
            float currIntensity = dctBlock.at<float>(r, c);
            quantBlock.at<float>(r, c) = round(currIntensity / quantisationMatrix.at<float>(r, c));
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
cv::Mat jpeg_block(cv::Mat block, int quantMatrixIndex, JpegElements &jpegElements) {
    // std::cout << "Init" << std::endl << block << std::endl << std::endl;

    // pre-process
    block.convertTo(block, CV_32F);
    // block -= 128;
    // std::cout << "Normalised" << std::endl << block << std::endl << std::endl;

    // dct
    cv::Mat dctBlock(8, 8, CV_32F);
    dct_block(dctBlock, block, jpegElements);
    // std::cout << "DCT'd" << std::endl << dctBlock << std::endl << std::endl;

    // quantise block
    cv::Mat quantBlock(8, 8, CV_32F);
    quantise_block(quantBlock, dctBlock, quantMatrixIndex, jpegElements);
    // std::cout << "Quantised" << std::endl << quantBlock << std::endl << std::endl;

    // // convert to flattened uchar array in zig-zag order
    // std::vector<int> block_array = block_to_zig_zag(quantBlock, jpegElements);
    // std::cout << "Zig-zag encoded" << std::endl;
    // PrintUtils::print_vector(block_array);

    // inverse dct
    cv::Mat invDctBlock(8, 8, CV_32F);
    inverse_dct_block(invDctBlock, quantBlock, jpegElements);
    // std::cout << "Inverted" << std::endl << invDctBlock << std::endl << std::endl;

    // de-normalised
    // invDctBlock += 128;
    // std::cout << "De-normalised" << std::endl << invDctBlock << std::endl << std::endl;

    // convert back to uchar
    cv::Mat finalBlock;
    invDctBlock.convertTo(finalBlock, CV_8UC1);

    // huffman encode
    // Huffman h;
    // std::vector<uchar> huff_encoded_data = h.encode_data(block_array);
    // std::cout << std::endl << "Final byte array: (" << huff_encoded_data.size() << ")" << std::endl;
    // PrintUtils::print_vector(huff_encoded_data);

    return finalBlock;
}

int jpeg(int imageNum, int quantMatrixIndex) {
    JpegElements jpegElements = JpegElements();

    // load image
    // std::string filename = "images/test_2.png";
    std::string filename = getDiv2kFileName(imageNum);
    cv::Mat image = CvImageUtils::load_image(filename);
    CvImageUtils::display_image(image, std::to_string(imageNum));

    // pad to make dimensions multiple of BLOCK_SIZE
    cv::Mat paddedImage = pad_for_jpeg(image, 8);

    // convert to Y, Cr, Cb format
    cv::Mat ycbcrImage = bgr_to_ycbcr(paddedImage);

    // split into channels
    std::vector<cv::Mat> channels;
    split(ycbcrImage, channels);
    
    int M = ycbcrImage.rows, N = ycbcrImage.cols;
    int blockNum = 0;
    cv::Mat currChannel, block, invDctBlock;

    std::vector<cv::Mat> invChannels;

    for (int channel = 0; channel < 3; channel++) {
        currChannel = channels[channel];
        cv::Mat invChannel = cv::Mat(M, N, CV_8UC1, cv::Scalar(0));
        for (int r = 0; r < M; r+=8) {
            for (int c = 0; c < N; c+=8) {
                cv::Rect blockRect(c, r, 8, 8);
                block = currChannel(blockRect).clone();
                invDctBlock = jpeg_block(block, quantMatrixIndex, jpegElements);
                // std::cout << block << std::endl;
                // std::cout << invDctBlock << std::endl << std::endl;
                invDctBlock.copyTo(invChannel(blockRect));
                blockNum++;
            }
        }
        invChannels.push_back(invChannel);
    }

    cv::Mat reconstructedImage;
    merge(invChannels, reconstructedImage);

    cv::Mat finalImage = ycbcr_to_bgr(reconstructedImage);
    CvImageUtils::display_image(finalImage, std::to_string(imageNum) + std::to_string(quantMatrixIndex));

    cv::destroyAllWindows();

    // TODO: concat block arrays to produce channel encodings
    // TODO: concat channel encodings to produce final data
    return 0;
}

void experiment();
void test_huffman_tree_build();
void test_dct_inverse();
void test_ycbcr();

int main() {
    int qmi = 3;
    int imageNum;
    for (imageNum = 1; imageNum < 10; imageNum++) {
        jpeg(imageNum, qmi);
    }
    
    return 0;
}

//// TESTING ////

void experiment() {
    Experiments exp;
    exp.blacken_pixels();
    exp.remove_pixels();
    exp.avg_pool();
    exp.max_pool();
}

void test_huffman_tree_build() {
    Huffman h;
    std::vector<int> data = {1,1,2,2,3,3,4,4};
    std::vector<uchar> enc_data = h.encode_data(data);
}

void test_dct_inverse() {
    JpegElements jpegElements = JpegElements();

    // // initialise 8x8 block of 255
    // cv::Mat block(8, 8, CV_32F, cv::Scalar(255));
    // std::cout << block << std::endl << std::endl;

    // initialise 8x8 random block
    cv::Mat block(8, 8, CV_32F);
    cv::randu(block, 0, 256);
    std::cout << block << std::endl << std::endl;

    cv::Mat dctBlock(8, 8, CV_32F);
    dct_block(dctBlock, block, jpegElements);
    std::cout << dctBlock << std::endl << std::endl;

    // inverse dct
    cv::Mat invDctBlock(8, 8, CV_32F);
    inverse_dct_block(invDctBlock, dctBlock, jpegElements);
    std::cout << invDctBlock << std::endl;
}

void test_ycbcr() {
    std::string filename = "images/test_2.png";
    cv::Mat image = CvImageUtils::load_image(filename);
    CvImageUtils::display_image(image, "");

    // cv::Mat ycbcrImage;
    // cv::cvtColor(image, ycbcrImage, cv::COLOR_YCrCb2BGR);
    cv::Mat ycbcrImage = bgr_to_ycbcr(image);
    CvImageUtils::display_image(ycbcrImage, "");

    // cv::Mat bgrImage;
    // cv::cvtColor(ycbcrImage, bgrImage, cv::COLOR_BGR2YCrCb);
    cv::Mat bgrImage = ycbcr_to_bgr(ycbcrImage);
    CvImageUtils::display_image(bgrImage, "");
}