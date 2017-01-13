//since threshold has a gpu and cpu version, may as well have a way to not setup cuda if we only want the cpu version
#define GPU_VER

#ifdef GPU_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

__global__ void thresholdKernel(unsigned char* original, unsigned char* modified, unsigned char threshold)
{
	int current = blockIdx.x * gridDim.x + threadIdx.x;
	if (original[current] > threshold) {
		modified[current] = 255;
	}
	else {
		modified[current] = 0;
	}
}

void threshold(unsigned char threshold, Mat& image);
cudaError_t GPUThreshold(unsigned char threshold, Mat& original);

int main(int argc, char* argv[])
{
	if (argc != 2) {
		printf("Usage: %s ImageToLoadAndDisplay\n", argv[0]);
		exit(1);
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	printf("Number of channels: %d\n", image.channels());

	if (!image.data) {
		printf("Could not find or open the image\n");
		exit(1);
	}

	cvtColor(image, image, COLOR_RGB2GRAY);

	unsigned char THRESHOLD = 100;

#ifdef GPU_VER
	cudaError_t cudaStatus;

	cudaStatus = GPUThreshold(THRESHOLD, image);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to apply threshold filter\n");
		//not sure what state the image will be in after things fail, but it's probably better to just stop
		exit(1);
	}
#endif
#ifndef GPU_VER
	threshold(THRESHOLD, image);
#endif

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);

	waitKey(0);

#ifdef GPU_VER
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED on cudaDeviceReset\n");
		exit(1);
	}
#endif
	return 0;
}

void threshold(unsigned char threshold, Mat& image)
{
	unsigned char* end_data = image.data + (image.rows * image.cols);
	for (unsigned char* p = image.data; p < end_data; p++) {
		if (*p > threshold) {
			*p = 255;
		}
		else {
			*p = 0;
		}
	}
}

cudaError_t GPUThreshold(unsigned char threshold, Mat& original)
{
	unsigned char* dev_original = 0;
	unsigned char* dev_modified = 0;
	int* dev_threshold = 0;
	cudaError_t cudaStatus;
	int dataSize = original.rows * original.cols * sizeof(unsigned char);
	try
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to set CUDA device";
		}

		cudaStatus = cudaMalloc((void**)&dev_original, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_original";
		}

		cudaStatus = cudaMalloc((void**)&dev_modified, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_modified";
		}

		cudaStatus = cudaMalloc((void**)&dev_threshold, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_threshold";
		}

		cudaStatus = cudaMemcpy(dev_original, original.data, dataSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy image data to GPU memory";
		}

		int numBlocks = original.rows * original.cols / 1024;
		thresholdKernel<<<numBlocks, 1024>>>(dev_original, dev_modified, threshold);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "FAILED launching thresholdKernel: %s\n", cudaGetErrorString(cudaStatus));
			throw " ";
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "ERROR after launching thresholdKernel: %d\n", cudaStatus);
			throw " ";
		}
	}
	catch (char* e) {
		fprintf(stderr, "%s\n", e);
	}

	//free memory (might move into catch block)
	if (dev_original != 0) {
		cudaFree(dev_original);
	}
	if (dev_modified != 0) {
		cudaFree(dev_modified);
	}

	return cudaStatus;
}