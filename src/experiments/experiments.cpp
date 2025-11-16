#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <algorithm>

#include "shared.hpp"
#include "experiments.hpp"

#define DISPLAY_SIZE 1000

Experiments::Experiments(std::string imageFilePath) {
    this->imageFilePath = imageFilePath;
    this->image = CvImageUtils::loadImage(imageFilePath);
    this->N = this->image.rows, this->M = this->image.cols;
    CvImageUtils::printImageStats(this->image);
}

/**
 * Randomly make fixed percentage of pixels black
 */
void Experiments::blacken_pixels() {
    CvImageUtils::displayImage(this->image, "Before");

    int perc = 70;
    cv::Mat workingImage = this->image.clone();
    int totalPixels = this->N * this->M;
    int numPixelsToBlack = (perc * totalPixels) / 100;
    
    // generate list of all pixel indices
    std::vector<int> allPixels(totalPixels);
    std::iota(allPixels.begin(), allPixels.end(), 0);
    
    // randomly permute list 
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(allPixels.begin(), allPixels.end(), g);
    
    // Pick the first `numPixelsToBlack` indices
    for (int i = 0; i < numPixelsToBlack; ++i) {
        int index = allPixels[i];
        int r = index / this->M;
        int c = index % this->M;
        workingImage.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
    }
    
    CvImageUtils::displayImage(workingImage, "After");
    cv::destroyAllWindows();
}

/**
 * Systematically remove fixed percentage of pixels from each row,
 * then re-construct image
 */
void Experiments::remove_pixels() {
    CvImageUtils::displayImage(this->image, "Before");

    int everyN = 10; // everyN pixels, include one
    int count;

    cv::Mat workingImage = this->image.clone();
    cv::Mat newImage;

    count = 0;
    for (int r = 0; r < workingImage.rows; ++r) {
        std::vector<cv::Vec3b> newRow;
        for (int c = 0; c < workingImage.cols; ++c) {
            if (count % everyN == 0) {
                newRow.push_back(workingImage.at<cv::Vec3b>(r, c));
            }
            count++;
        }
        
        // add new row to image
        cv::Mat newRowCv(newRow, true);
        newRowCv = newRowCv.reshape(3, 1);
        if (newImage.empty()) {
            newImage = newRowCv;
        } else {
            cv::vconcat(newImage, newRowCv, newImage);
        }
    }

    CvImageUtils::printImageStats(newImage);

    cv::Mat resizedImage;
    cv::resize(newImage, resizedImage, cv::Size(this->image.cols, this->image.rows));
    CvImageUtils::displayImage(resizedImage, "After");

    cv::destroyAllWindows();
}

/**
 * Avg pooling
 */
void Experiments::avg_pool() {
    CvImageUtils::displayImage(this->image, "Before");

    int kern_size = 3;
    int newRows = this->image.rows / kern_size;
    int newCols = this->image.cols / kern_size;
    cv::Mat pooledImage(newRows, newCols, this->image.type());

    for (int r = 0; r < newRows; r++) {
        for (int c = 0; c < newCols; c++) {
            int rStart = r * kern_size;
            int cStart = c * kern_size;
            cv::Vec3b meanVals;
            for (int channel = 0; channel < 3; channel++)  {
                float sum = 0;
                for (int i = 0; i < kern_size; i++) {
                    for (int j = 0; j < kern_size; j++) {
                        sum += image.at<cv::Vec3b>(rStart + i, cStart + j)[channel];
                    }
                }
                uchar meanVal = static_cast<uchar>(sum / (kern_size * kern_size));
                meanVals[channel] = meanVal;
            }
            pooledImage.at<cv::Vec3b>(r, c) = meanVals;
        }
    }

    cv::Mat resizedImage;
    cv::resize(pooledImage, resizedImage, cv::Size(this->image.cols, this->image.rows));
    CvImageUtils::displayImage(resizedImage, "After");

    cv::destroyAllWindows();
}

/**
 * Max pooling
 */
void Experiments::max_pool() {
    CvImageUtils::displayImage(this->image, "Before");

    int kern_size = 3;
    int newRows = this->image.rows / kern_size;
    int newCols = this->image.cols / kern_size;
    cv::Mat pooledImage(newRows, newCols, this->image.type());

    for (int r = 0; r < newRows; r++) {
        for (int c = 0; c < newCols; c++) {
            int rStart = r * kern_size;
            int cStart = c * kern_size;
            cv::Vec3b maxVals;
            for (int channel = 0; channel < 3; channel++)  {
                uchar maxVal = 0;
                for (int i = 0; i < kern_size; i++) {
                    for (int j = 0; j < kern_size; j++) {
                        maxVal = std::max(maxVal, image.at<cv::Vec3b>(rStart+i, cStart+j)[channel]);
                    }
                }
                maxVals[channel] = static_cast<uchar>(maxVal);
            }
            pooledImage.at<cv::Vec3b>(r, c) = maxVals;
        }
    }

    cv::Mat resizedImage;
    cv::resize(pooledImage, resizedImage, cv::Size(this->image.cols, this->image.rows));
    CvImageUtils::displayImage(resizedImage, "After");

    cv::destroyAllWindows();
}