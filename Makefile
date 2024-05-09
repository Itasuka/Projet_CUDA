CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
LDLIBS1=`pkg-config --libs opencv4`
NVFLAGS=-O3 -I/usr/include/opencv4/ -ccbin g++-10
LDLIBS2=-lm -lIL

all: laplacian_operator-cu laplacian_operator gaussian-cu gaussian sobel-cu sobel box-cu box

laplacian_operator-cu: laplacian_operator.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)
	
laplacian_operator: laplacian_operator.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)
	
gaussian-cu: gaussian.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)
	
gaussian: gaussian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)
	
sobel-cu: sobel.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)
	
box-cu: boxblur.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

box: boxblur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

.PHONY: clean

clean:
	rm laplacian_operato-cu laplacian_operator gaussian-cu gaussian sobel-cu sobel box-cu box
