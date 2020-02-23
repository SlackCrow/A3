CXX=nvcc
CXXFLAGS=-x cu -std=c++11 -O3

all: a3

clean:
	rm -rf a3
