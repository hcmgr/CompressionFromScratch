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

/**
 * Run-length-encoding of given block array
 */
int rle(std::vector<uchar>) {
    return 0;
}

int muckin() {
    // Load the image
    std::string image_name = "test_images/test_1.png";
    cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    // split into channels
    std::vector<cv::Mat> channels;
    split(image, channels);
    cv::Mat blue_channel = channels[1];

    // extracting 8x8 block
    cv::Rect roi_rect(0, 0, 8, 8);
    cv::Mat block = blue_channel(roi_rect).clone();
    std::cout << block << std::endl;

    // flatten and convert to vector<uchar>
    cv::Mat flattenedBlock = block.reshape(0, 1).clone();
    uchar* data = flattenedBlock.ptr();
    std::vector<uchar> flattenedVec(data, data + flattenedBlock.cols);
    for (uchar el : flattenedVec) {
        std::cout << static_cast<int>(el) << std::endl;
    }
    std::cout << flattenedVec.size() << std::endl;

    display_image(image, image_name);
    return 0;
}

int main() {
    muckin();
    return 0;
}


