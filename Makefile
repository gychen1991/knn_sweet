NVCC=nvcc -w -arch=sm_35 -std=c++11 -pg -lcublas -O3 -Xcompiler -fopenmp -I./include

all: knnjoin
knnjoin: ./src/knnjoin.cu
	$(NVCC) -o $@ $^
clean:
	rm knnjoin *.out
