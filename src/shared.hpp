#pragma once

#include <opencv2/opencv.hpp>

#pragma once

#include <opencv2/opencv.hpp>

//
// A collection of OpenCV utility functions
//
namespace CvImageUtils {

    //
    // Load image from file into a cv::Mat structure
    //
    cv::Mat loadImage(std::string filename);

    //
    // Save the given cv::Mat image to the specified file path.
    // Returns true on success, false on failure.
    //
    bool saveImage(const cv::Mat& image, const std::string& path);

    //
    // Displays the given cv::Mat image in a window of the given name
    //
    void displayImage(cv::Mat& image, std::string name);

    //
    // Pretty prints various cv::Mat image statistics
    //
    void printImageStats(cv::Mat& image);
};

//
// Collection of utility functions to print data structures nicely
//
namespace PrintUtils {

    //
    // Pretty-print std::vector<T>
    //
    template<typename T>
    void printVector(const std::vector<T>& vec) {
        std::cout << "[ ";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << static_cast<int>(vec[i]);
            if (i != vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
    }

    //
    // Pretty-print std::map<K, V>
    //
    template <typename K, typename V>
    void printMap(const std::map<K, V>& m) {
        std::cout << "{\n";
        for (const auto& pair : m) {
            std::cout << "  " << pair.first << ": " << pair.second << "\n";
        }
        std::cout << "}\n";
    }
};

//
// Collection of math utility functions
//
namespace MathUtils {

    //
    // Clamp 'value' into range ['min', 'max']
    //
    int clamp(int value, int min, int max);
}
