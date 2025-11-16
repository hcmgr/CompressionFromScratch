#pragma once

#include <vector>

//
// Run-length encoding/decoding algorithms
//
namespace Rle {

    //
    // Run-length encode given byte array
    //
    std::vector<int> rleEncode(std::vector<int> arr);
}