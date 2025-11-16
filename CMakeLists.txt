cmake_minimum_required(VERSION 3.10)
project(jpeg CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# OpenCV (Homebrew) 
# CMake will look under /opt/homebrew automatically if you set CMAKE_PREFIX_PATH
find_package(OpenCV REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_executable(myjpeg ${SOURCES})
target_include_directories(myjpeg PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(myjpeg PRIVATE ${OpenCV_LIBS})
install(TARGETS myjpeg DESTINATION bin)