#include <opencv2/opencv.hpp>
#include "shared.hpp"

/**
 * A Collection of OpenCV utility functions
 */
namespace CvImageUtils {

    /**
     * Load image from file into a cv::Mat structure
     */
    cv::Mat load_image(std::string filename) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Could not open or find the image: " << filename << std::endl;
        }
        return image;
    }

    /**
     * Displays the given cv::Mat image in a window of the given name
     */
    void display_image(cv::Mat& image, std::string name) {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        // cv::resizeWindow(name, image.cols / 2, image.rows / 2);
        cv::imshow(name, image);
        cv::waitKey(0);
    }

    /**
     * Pretty prints various cv::Mat image statistics
     */
    void print_image_stats(cv::Mat& image) {
        std::cout << "NROWS: " << image.rows << std::endl;
        std::cout << "NCOLS: " << image.cols << std::endl;
        std::cout << "NCHANNELS: " << image.channels() << std::endl;
    }
};

namespace MathUtils {

    /**
     * Clamp 'value' into range ['min', 'max']
     */
    int clamp(int value, int min, int max) {
        return std::max(min, std::min(value, max));
    }
}