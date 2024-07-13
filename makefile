CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
OPENCV_FLAGS = `pkg-config --libs --cflags opencv4`
SRCDIR = src
OBJDIR = obj
TARGET = jpeg

SOURCES = $(wildcard $(SRCDIR)/*.cpp)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) *.out