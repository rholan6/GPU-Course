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

/*
 *	Global Variables
 */
unsigned char thresholdSlider = 128;
char* windowName = "Display window";
unsigned char* dev_original = 0;
unsigned char* dev_modified = 0;
int dataSize = 0;
Mat image;

/*
 *	Threshold Kernel
 */
__global__ void thresholdKernel(unsigned char* original, unsigned char* modified, unsigned char threshold, int size)
{
	int current = blockIdx.x * /*gridDim.x*/blockDim.x + threadIdx.x;
	if (current < size)
	{
		if (original[current] > threshold) {
			modified[current] = 255;
		}
		else {
			modified[current] = 0;
		}
	}
	/*unsigned char* current = original + (blockIdx.x * gridDim.x + threadIdx.x);
	unsigned char* modifiedCurrent = modified + (blockIdx.x * gridDim.x + threadIdx.x);
	if (*current > *threshold) {
		*modifiedCurrent = 255;
	}
	else {
		*modifiedCurrent = 0;
	}*/
}


/*
 *	Function prototypes
 */
void threshold(unsigned char threshold, Mat& image);
void on_trackbar(int, void*);
cudaError_t GPUThreshold(unsigned char threshold);


/*
 *	Main
 */
int main(int argc, char* argv[])
{
	if (argc != 2) {
		printf("Usage: %s ImageToLoadAndDisplay\n", argv[0]);
		exit(1);
	}

	//Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	printf("Number of channels: %d\n", image.channels());

	if (!image.data) {
		printf("Could not find or open the image\n");
		exit(1);
	}

	cvtColor(image, image, COLOR_RGB2GRAY);
	//namedWindow(windowName, WINDOW_NORMAL);
	//imshow(windowName, image);
	//waitKey(0);

	printf("Number of channels: %d\n", image.channels());

	//unsigned char THRESHOLD = 100;

#ifdef GPU_VER
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		throw "FAILED to set CUDA device";
	}

	//call this once so I don't need to move all the cuda setup code (and to see a thresholded image from the start)
	cudaStatus = GPUThreshold(thresholdSlider);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to apply threshold filter\n");
		//not sure what state the image will be in after things fail, but it's probably better to just stop
		exit(1);
	}

	namedWindow(windowName, WINDOW_NORMAL);
	createTrackbar("Threshold", windowName, (int*)&thresholdSlider, 255, on_trackbar);
	on_trackbar(thresholdSlider, 0);
	//imshow(windowName, image);
#endif
#ifndef GPU_VER
	threshold(THRESHOLD, image);
#endif

	//namedWindow("Display window", WINDOW_NORMAL);
	//imshow("Display window", image);

	waitKey(0);

#ifdef GPU_VER
	//free memory (might move into catch block)
	if (dev_original != 0) {
		cudaFree(dev_original);
	}
	if (dev_modified != 0) {
		cudaFree(dev_modified);
	}
	printf("GPU memory freed\n");

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED on cudaDeviceReset\n");
		exit(1);
	}
#endif
	return 0;
}

/*
 *	CPU Threshold
 */
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

/*
 *	Trackbar Handler
 */
void on_trackbar(int, void*)
{
	cudaError_t cudaStatus;
	printf("Threshold: %d\n", thresholdSlider);
	int numBlocks = (1023 + image.rows * image.cols) / 1024;
	thresholdKernel <<<numBlocks, 1024>>> (dev_original, dev_modified, thresholdSlider, dataSize);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR with threshold kernel: %d\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(image.data, dev_modified, dataSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to copy modified data to CPU memory\n");
	}
	imshow(windowName, image);
}

/*
 *	GPU Threshold
 */
cudaError_t GPUThreshold(unsigned char threshold)
{
	//unsigned char* dev_threshold = 0;
	cudaError_t cudaStatus;
	dataSize = image.rows * image.cols * sizeof(unsigned char);
	try
	{
		cudaStatus = cudaMalloc((void**)&dev_original, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_original";
		}

		cudaStatus = cudaMalloc((void**)&dev_modified, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_modified";
		}

		/*cudaStatus = cudaMalloc((void**)&dev_threshold, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_threshold";
		}*/

		cudaStatus = cudaMemcpy(dev_original, image.data, dataSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy image data to GPU memory";
		}

		/*cudaStatus = cudaMemcpy(dev_threshold, &threshold, sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy threshold to GPU memory";
		}*/
		
		printf("Setting numblocks\n");
		int numBlocks = (1023 + image.rows * image.cols) / 1024;
		printf("%d blocks\n", numBlocks);
		thresholdKernel<<<numBlocks, 1024>>>(dev_original, dev_modified, threshold, dataSize);

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

		cudaStatus = cudaMemcpy(image.data, dev_modified, dataSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy modified data to CPU memory";
		}
	}
	catch (char* e) {
		fprintf(stderr, "%s\n", e);
		//free memory (might move into catch block)
		if (dev_original != 0) {
			cudaFree(dev_original);
		}
		if (dev_modified != 0) {
			cudaFree(dev_modified);
		}
		printf("GPU memory freed\n");
	}

	/*//free memory (might move into catch block)
	if (dev_original != 0) {
		cudaFree(dev_original);
	}
	if (dev_modified != 0) {
		cudaFree(dev_modified);
	}
	printf("GPU memory freed\n");*/

	return cudaStatus;
}