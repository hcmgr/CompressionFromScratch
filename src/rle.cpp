#include <vector>

/**
 * Collection of run-length encoding/decoding algorithms
 */
namespace Rle {

    /**
     * Run-length encode given array
     */
    std::vector<int> rle_encode(std::vector<int> arr) {
        std::vector<int> rle_array;

        int n = arr.size();
        int i = 0, j = 0;
        while (i < n) {
            j = i+1;
            while (arr[j] == arr[i]) {
                j++;
            }
            rle_array.push_back((j - i)); // count
            rle_array.push_back((arr[i])); // number
            i = j;
        }

        return rle_array;
    }
}