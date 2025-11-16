#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

#include "shared.hpp"

//
// Self-contained experiments of different naive compression methods, including:
//      - blackend % of pixels
//      - remove % of pixels
//      - avg pooling
//      - max pooling
// Mostly used to show how glorious JPEG is.
//
class Experiments {
private:
    std::string imageFilePath;
    cv::Mat image; 
    int M, N;

public:
    Experiments(std::string imageFilePath); 

    //
    // Blacken x% of pixels
    //
    void blacken_pixels();

    //
    // Remove x% of pixels
    //
    void remove_pixels();

    //
    // Average pooling
    //
    void avg_pool();

    //
    // Max pooling
    //
    void max_pool();
};