# knn_sweet
@By Guoyang Chen(gychen1991@gmail.com)

Very Fast KNN implementation on GPU with Triangle Inequality Theory.

8x times faster than the state-of-art best:

http://nichol.as/papers/Garcia/Fast%20k%20nearest%20neighbor%20search%20using.pdf (Code: https://github.com/vincentfpgarcia/kNN-CUDA)

Our Paper is in submission...

Prerequisites:

1. NVIDIA GPU card installed on machine.
2. CUDA drivers and CUDA toolkit installed(See https://developer.nvidia.com/cuda-toolkit)

To compile:
	make

To Run:
	./run_sample.sh (find 200 nearest neighbors)
