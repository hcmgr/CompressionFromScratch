#include <opencv2/opencv.hpp>
#include "shared.hpp"

//
// A collection of OpenCV utility functions
//
namespace CvImageUtils {

    //
    // Load image from file into a cv::Mat structure
    //
    cv::Mat loadImage(std::string filename) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Could not open or find the image: " << filename << std::endl;
        }
        return image;
    }

    //
    // Save the given cv::Mat image to the specified file path.
    // Returns true on success, false on failure.
    //
    bool saveImage(const cv::Mat& image, const std::string& path) {
        if (image.empty()) {
            std::cout << "saveImage(): Cannot save an empty image to: " << path << std::endl;
            return false;
        }

        bool ok = cv::imwrite(path, image);
        if (!ok) {
            std::cout << "saveImage(): Failed to write image to: " << path << std::endl;
        }

        return ok;
    }

    //
    // Displays the given cv::Mat image in a window of the given name
    //
    void displayImage(cv::Mat& image, std::string name) {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::resizeWindow(name, image.cols / 1, image.rows / 1);
        cv::imshow(name, image);
        cv::waitKey(0);
    }

    //
    // Pretty prints various cv::Mat image statistics
    //
    void printImageStats(cv::Mat& image) {
        std::cout << "NROWS: " << image.rows << std::endl;
        std::cout << "NCOLS: " << image.cols << std::endl;
        std::cout << "NCHANNELS: " << image.channels() << std::endl;
    }
};

namespace MathUtils {

    //
    // Clamp 'value' into range ['min', 'max']
    //
    int clamp(int value, int min, int max) {
        return std::max(min, std::min(value, max));
    }
}
