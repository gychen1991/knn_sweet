# knn_sweet
@By [Guoyang Chen][1](gychen1991@gmail.com)

# Very Fast KNN implementation on GPU with Triangle Inequality Theory.

Our Paper is in submission...

# 8x times faster than the state-of-art best:
[Fast k nearest neighbor search using GPU][2] (Code: https://github.com/vincentfpgarcia/kNN-CUDA)

## Prerequisites:

1. NVIDIA GPU card installed on machine.
2. CUDA drivers and CUDA toolkit installed(See https://developer.nvidia.com/cuda-toolkit)

## To compile:
	make

## To Run:
	./run_sample.sh (find 200 nearest neighbors)
	
[1]:http://research.csc.ncsu.edu/nc-caps/gchen11/index.html
[2]:http://nichol.as/papers/Garcia/Fast%20k%20nearest%20neighbor%20search%20using.pdf
