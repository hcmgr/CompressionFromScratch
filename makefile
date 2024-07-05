CC = g++
OPENCV_FLAGS = `pkg-config --libs --cflags opencv4`

comp: comp.cpp
	$(CC) $(CFLAGS) -o comp comp.cpp $(OPENCV_FLAGS)

clean:
	rm -f comp *.out
