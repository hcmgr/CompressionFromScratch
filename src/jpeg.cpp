#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

#include "pre_computed.hpp"
#include "huffman.hpp"
#include "shared.hpp"
#include "rle.hpp"
#include "experiments.hpp"

//
// Pads given image to ensure its dimensions are a multiple of 'blockSize'
//
cv::Mat padForJpeg(cv::Mat image, int blockSize) {
    int padRows = blockSize - (image.rows % blockSize);
    int padCols = blockSize - (image.cols % blockSize);
    if (padRows == blockSize) padRows = 0;
    if (padCols == blockSize) padCols = 0;

    cv::Mat paddedImage;
    cv::copyMakeBorder(image, paddedImage, 0, padRows, 0, padCols, cv::BORDER_REPLICATE);
    return paddedImage;
}

//
// Convert image from [B,G,R] to [Y,Cb,Cr] format
//
cv::Mat bgrToYcbcr(cv::Mat bgrImage) {
    cv::Mat ycbcrImage(bgrImage.rows, bgrImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    float b, g, r;
    float y, cb, cr;
    for (int i = 0; i < bgrImage.rows; i++) {
        for (int j = 0; j < bgrImage.cols; j++) {
            cv::Vec3b &bgrPixels = bgrImage.at<cv::Vec3b>(i, j);
            b = bgrPixels[0]; 
            g = bgrPixels[1]; 
            r = bgrPixels[2];

            // coefficient voodoo
            y = 0.299 * r + 0.587 * g + 0.114 * b;
            cb = 128 + 0.5*b - 0.168736*r - 0.331364*g;
            cr = 128 + 0.5*r - 0.418688*g - 0.081312*b;

            cv::Vec3b &ycbcrPixels = ycbcrImage.at<cv::Vec3b>(i, j);
            ycbcrPixels[0] = MathUtils::clamp(round(y), 0, 255); // Y
            ycbcrPixels[1] = MathUtils::clamp(round(cr), 0, 255); // Cr
            ycbcrPixels[2] = MathUtils::clamp(round(cb), 0, 255); // Cb
        }
    }
    return ycbcrImage;
}

//
// Convert image from [Y, Cb, Cr] to [B,G,R] format
//
cv::Mat ycbcrToBgr(cv::Mat ycbcrImage) {
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

            // more coefficient voodoo
            r = y + 1.402*cr;
            g = y - 0.344136*cb - 0.714136*cr;
            b = y + 1.772*cb;

            cv::Vec3b &bgrPixels = bgrImage.at<cv::Vec3b>(i, j);
            bgrPixels[0] = MathUtils::clamp(round(b), 0, 255); // b
            bgrPixels[1] = MathUtils::clamp(round(g), 0, 255); // g
            bgrPixels[2] = MathUtils::clamp(round(r), 0, 255); // r
        }
    }
    return bgrImage;
}

//
// Performs DCT step on the given 8x8 block
//
void dctBlock(cv::Mat dctBlock, cv::Mat block, JpegElements &jpegElements) {
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

//
// Performs the inverse DCT step on the given 8x8 block
//
void inverseDctBlock(cv::Mat invBlock, cv::Mat dctBlock, JpegElements &jpegElements) {
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

//
// Performs quantisation step on the given 8x8 block
//
void quantiseBlock(cv::Mat quantBlock, cv::Mat dctBlock, int i, JpegElements &jpegElements) {
    cv::Mat quantisationMatrix = jpegElements.getQuantisationMatrix(i);
    for (int r = 0; r < BLOCK_SIZE; r++) {
        for (int c = 0; c < BLOCK_SIZE; c++) {
            float currIntensity = dctBlock.at<float>(r, c);
            quantBlock.at<float>(r, c) = round(currIntensity / quantisationMatrix.at<float>(r, c));
        }
    }
}

//
// Convert 8x8 block into zig-zag ordered 64-d array
//
std::vector<int> blockToZigZag(cv::Mat block, JpegElements &jpegElements) {
    std::vector<int> res;

    if (jpegElements.zig_zag_indices.size() != BLOCK_SIZE*BLOCK_SIZE) {
        std::cout << "Fatal: incorrect block size" << "\n";
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

//
// For given block, apply jpeg, then reverse steps to produce compressed block
//
cv::Mat jpegBlockForwardReverse(cv::Mat block, int quantMatrixIndex, JpegElements &jpegElements, bool debug) {
    // pre-process
    cv::Mat floatBlock;
    block.convertTo(floatBlock, CV_32F);

    // dct
    cv::Mat dctResultBlock(8, 8, CV_32F);
    dctBlock(dctResultBlock, floatBlock, jpegElements);
    
    // quantise block
    cv::Mat quantBlock(8, 8, CV_32F);
    quantiseBlock(quantBlock, dctResultBlock, quantMatrixIndex, jpegElements);
    
    // inverse dct
    cv::Mat invDctBlock(8, 8, CV_32F);
    inverseDctBlock(invDctBlock, quantBlock, jpegElements);
    
    // convert back to uchar
    cv::Mat finalBlock;
    invDctBlock.convertTo(finalBlock, CV_8UC1);
    
    if (debug) {
        std::cout << "Init" << "\n" << block << "\n" << "\n";
        std::cout << "DCT'd" << "\n" << dctResultBlock << "\n" << "\n";
        std::cout << "Quantised" << "\n" << quantBlock << "\n" << "\n";
        std::cout << "Converted back" << "\n" << finalBlock << "\n" << "\n";
    }

    return finalBlock;
}

//
// Apply jpeg to image, then reverse it and re-construct compressed form.
//
int jpegForwardReverse(std::string imageFilePath, int quantMatrixIndex) {
    JpegElements jpegElements = JpegElements();

    // load image
    cv::Mat image = CvImageUtils::loadImage(imageFilePath);
    std::cout << "loaded image: " << imageFilePath << "\n";

    CvImageUtils::displayImage(image, "Before (" + imageFilePath + ")");

    // pad to make dimensions multiple of BLOCK_SIZE
    cv::Mat paddedImage = padForJpeg(image, 8);

    // convert to Y, Cr, Cb format
    cv::Mat ycbcrImage = bgrToYcbcr(paddedImage);

    // split into channels
    std::vector<cv::Mat> channels;
    split(ycbcrImage, channels);
    
    int M = ycbcrImage.rows, N = ycbcrImage.cols, nChannels = 3;
    cv::Mat currChannel, block, invDctBlock;

    std::vector<cv::Mat> invChannels;

    for (int channel = 0; channel < nChannels; channel++) {
        currChannel = channels[channel];
        cv::Mat invChannel = cv::Mat(M, N, CV_8UC1, cv::Scalar(0));
        for (int r = 0; r < M; r+=8) {
            for (int c = 0; c < N; c+=8) {
                cv::Rect blockRect(c, r, 8, 8);
                block = currChannel(blockRect).clone();
                invDctBlock = jpegBlockForwardReverse(block, quantMatrixIndex, jpegElements, false);
                invDctBlock.copyTo(invChannel(blockRect));
            }
        }
        invChannels.push_back(invChannel);
    }

    // reconstruct and display final image
    cv::Mat reconstructedImage;
    merge(invChannels, reconstructedImage);
    cv::Mat finalImage = ycbcrToBgr(reconstructedImage);
    CvImageUtils::displayImage(finalImage, "After (" + imageFilePath + ")");

    // cleanup
    cv::destroyAllWindows();
    return 0;
}

////////////////////////////////////////
// Run
////////////////////////////////////////

struct CliArgs {
    std::string imagePath;      

    // quantisation matrix to use - default 3 (best performing so far)
    int qmi = 3;           
};

std::string usage() {
    std::ostringstream oss;
    oss << "Usage: ./jpeg {image_path} [--qmi=N]" << "\n\n";
    oss << "Note - valid N values: {0,1,2,3,4} (increasing orders of quantisation)" << "\n";
    return oss.str();
}

CliArgs parseCliArgs(int argc, char* argv[]) {
    CliArgs args;

    if (argc < 2) {
        std::cout << usage();
        std::exit(1);
    }

    args.imagePath = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        int qmi;
        if (arg.rfind("--qmi=", 0) == 0) {
            qmi = std::stoi(arg.substr(6));
        } else {
            std::cout << usage();
            std::exit(1);
        }
        if (qmi < 0 || qmi > NUM_QUANT_MATRICES) {
            std::cout << usage();
            std::exit(1);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    CliArgs args = parseCliArgs(argc, argv);
    jpegForwardReverse(args.imagePath, args.qmi);
    return 0;
}