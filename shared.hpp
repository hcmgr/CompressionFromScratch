#ifndef SHARED_H
#define SHARED_H

#include <opencv2/opencv.hpp>

/**
 * A Collection of OpenCV utility functions
 */
namespace CvImageUtils {

    /**
     * Load image from file into a cv::Mat structure
     */
    cv::Mat loadImage(std::string filename);

    /**
     * Displays the given cv::Mat image in a window of the given name
     */
    void display_image(cv::Mat& image, std::string name);

    /**
     * Pretty prints various cv::Mat image statistics
     */
    void print_image_stats(cv::Mat& image);
}

#endif // SHARED_H