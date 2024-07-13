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
};

/**
 * Collection of utilty functions to print data structures nicely
 */
namespace PrintUtils {

    /**
     * Pretty-print std::vector<T>
     */
    template<typename T>
    void print_vector(const std::vector<T>& vec) {
        std::cout << "[ ";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << static_cast<int>(vec[i]);
            if (i != vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
    }

    /**
     * Pretty-print std::map<K, V>
     */
    template <typename K, typename V>
    void print_map(const std::map<K, V>& m) {
        std::cout << "{\n";
        for (const auto& pair : m) {
            std::cout << "  " << pair.first << ": " << pair.second << "\n";
        }
        std::cout << "}\n";
    }
};

#endif // SHARED_H