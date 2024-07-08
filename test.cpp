#include <iostream>

int main() {
    std::string s = "1100010";
    int v = std::stoi(s, nullptr, 2);
    std::cout << v << std::endl;
    unsigned char c = static_cast<unsigned char>(v);
    std::cout << c << std::endl;
    return 0;
}
