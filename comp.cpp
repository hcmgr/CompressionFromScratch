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
 * Compute and return indices of 8x8 zig zag traversal
 */
std::vector<std::pair<int,int>> zig_zag_indices() {
    std::vector<std::pair<int, int>> indices;
    indices.push_back(std::make_pair(0, 0));

    int cnt = 1;
    int r = 0, c = 0;
    int up = 1;
    while (cnt < 64) {
        // top side
        if (r == 0) {
            if (up) {
                c += 1; // across 1
                up = 0;
            } else {
                c -= 1; r += 1; // down-left 1
            }
        }

        // bottom side
        else if (r == 7) {
            if (up) {
                c += 1; r -= 1; // up-right 1
            } else {
                c += 1; // across 1
                up = 1;
            }
        }

        // left side
        else if (c == 0) {
            if (up) {
                c += 1; r -= 1; // up-right 1
            } else {
                r += 1; // down 1
                up = 1;
            }
        }

        // right side
        else if (c == 7) {
            if (up) {
                r += 1; // down 1
                up = 0;
            } else {
                c -= 1; r += 1; // down-left 1
            }
            
        }

        // other
        else {
            if (up) {
                c += 1; r -= 1; // up-right 1
            } else {
                c -= 1; r += 1; // down-left 1
            }
        }

        indices.push_back(std::make_pair(r, c));
        cnt++;
    }

    return indices;
}

/**
 * Run-length-encoding of given block array
 */
std::vector<uchar> rle(std::vector<uchar> block_array) {
    std::vector<uchar> rle_array;

    int n = block_array.size();
    int i = 0, j = 0;
    while (i < n) {
        j = i+1;
        while (block_array[j] == block_array[i]) {
            j++;
        }
        rle_array.push_back((j - 1)); // count
        rle_array.push_back((block_array[i])); // number
        i = j;
    }

    return rle_array;
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
    cv::Rect roi_rect(150, 150, 8, 8);
    cv::Mat block = blue_channel(roi_rect).clone();

    // flatten and convert to vector<uchar>
    cv::Mat flattenedBlock = block.reshape(0, 1).clone();
    uchar* data = flattenedBlock.ptr();
    std::vector<uchar> flattenedVec(data, data + flattenedBlock.cols);
    // for (uchar el : flattenedVec) {
    //     std::cout << static_cast<int>(el) << std::endl;
    // }
    // std::cout << "-------------" << std::endl;

    // rle encode vector
    std::vector<uchar> rleVec = rle(flattenedVec);
    // for (uchar el : rleVec) {
    //     std::cout << static_cast<int>(el) << std::endl;
    // }

    // print zig zag indices
    std::vector<std::pair<int, int>> inds = zig_zag_indices();
    for (std::pair<int, int> rc : inds) {
        std::cout << rc.first << ", " << rc.second << std::endl;
    }

    return 0;
}

int main() {
    muckin();
    return 0;
}


