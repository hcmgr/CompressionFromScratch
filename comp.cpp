#include <opencv2/opencv.hpp>
#include <iostream>
#include <typeinfo>

void print_image_stats(cv::Mat& image) {
    std::cout << "NROWS: " << image.rows << std::endl;
    std::cout << "NCOLS: " << image.cols << std::endl;
    std::cout << "NCHANNELS: " << image.channels() << std::endl;
}

void display_image(cv::Mat& image, std::string name) {
    print_image_stats(image);
    cv::imshow(name, image);
    cv::waitKey(0);
}

int muckin() {
    int width = 640;
    int height = 480;

    // Create a grayscale image (1 channel)
    cv::Mat image(height, width, CV_8UC1);

    // Generate black to white gradient
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Calculate intensity based on position
            uchar intensity = static_cast<uchar>((x + y) * 255 / (image.cols + image.rows));
            image.at<uchar>(y, x) = intensity;
        }
    }

    display_image(image, "Gradient");
    return 0;
}

int main() {
    // Load the image
    std::string image_name = "test_images/test_1.png";
    cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);

    std::vector<uchar> vec;
    cv::Vec3b pixel;
    pixel = image.at<cv::Vec3b>(349, 20);
    std::cout << pixel << std::endl;
    int r = image.rows - 1;
    int n = image.cols;
    for (int c = 0; c < n; c++) {
        pixel = image.at<cv::Vec3b>(r, c);
        vec.push_back(pixel[1]);
    }

    for (uchar el : vec) {
        std::cout << static_cast<int>(el) << std::endl;
    }

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    display_image(image, image_name);
    // muckin();
    return 0;
}


