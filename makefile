CC = g++
OPENCV_FLAGS = `pkg-config --libs --cflags opencv4`
SRCS = jpeg.cpp pre_computed.cpp huffman.cpp shared.cpp rle.cpp
EXEC = jpeg

jpeg: $(SRCS)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRCS) $(OPENCV_FLAGS)

clean:
	rm -f $(EXEC) *.out
