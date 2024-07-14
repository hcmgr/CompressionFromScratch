#include <opencv2/opencv.hpp>
#include <iostream>

#include "shared.hpp"

/**
 * Collection of self-contained experiments
 */
class Experiments {
public:
    Experiments(); 

    /**
     * Remove x% of pixels
     */
    void blacken_pixels();

    void remove_pixels();

    /**
     * Average pooling
     */
    void avg_pool();

    /**
     * Max pooling
     */
    void max_pool();

private:
    std::string filename;
    cv::Mat image; 
    int M, N;
};