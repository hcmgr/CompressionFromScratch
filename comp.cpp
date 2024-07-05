#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat image = cv::imread("dwarkesh.jpg", cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Create a window for display
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    // Show the image inside the window
    cv::imshow("Display window", image);

    // Wait for a keystroke in the window
    cv::waitKey(0);
    return 0;
}


