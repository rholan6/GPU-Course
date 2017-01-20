//since threshold has a gpu and cpu version, may as well have a way to not setup cuda if we only want the cpu version
//#define GPU_VER

#ifdef GPU_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "hpt.h"

using namespace cv;

typedef unsigned char UBYTE;

/*
 *	Global Variables
 */
UBYTE thresholdSlider = 128;
//GPU pointers
UBYTE* dev_original = 0;
UBYTE* dev_modified = 0;
//general
int dataSize = 0;
Mat image;
char* windowName = "Display window";
//timer
HighPrecisionTime hpt;
double totalTime = 0.0;
int runs = 0;
//kernel for the box filter
//edgy
//int boxKernel[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
//3x3 blur
//int boxKernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
//5x5 blur
//int boxKernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const int kw = 19;
const int kh = 19;
//generic blur
int boxKernel[kw * kh];

#ifdef GPU_VER
/*
 *	Threshold Kernel
 */
__global__ void thresholdKernel(UBYTE* original, UBYTE* modified, UBYTE threshold, int size)
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
	/*UBYTE* current = original + (blockIdx.x * gridDim.x + threadIdx.x);
	UBYTE* modifiedCurrent = modified + (blockIdx.x * gridDim.x + threadIdx.x);
	if (*current > *threshold) {
		*modifiedCurrent = 255;
	}
	else {
		*modifiedCurrent = 0;
	}*/
}
#endif


/*
 *	Function prototypes
 */
void threshold(UBYTE threshold, Mat& image);
#ifdef GPU_VER
void on_trackbar(int, void*);
cudaError_t GPUThreshold(UBYTE threshold);
#endif
void boxTrackbar(int, void*);
void boxFilter(UBYTE* src, UBYTE* dst, int w, int h, int* kernel, int kw, int kh, UBYTE* tmp);


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

	//printf("Number of channels: %d\n", image.channels());

	if (!image.data) {
		printf("Could not find or open the image\n");
		exit(1);
	}

	cvtColor(image, image, COLOR_RGB2GRAY);
	namedWindow(windowName, WINDOW_NORMAL);
	imshow(windowName, image);
	waitKey(0);

	//fill kernel with ones
	for (int i = 0; i < kw * kh; i++) {
		boxKernel[i] = 1;
	}

	//set up a trackbar for quick timing
	createTrackbar("Threshold", windowName, (int*)&thresholdSlider, 255, boxTrackbar);
	boxTrackbar(thresholdSlider, 0);

	//printf("Number of channels: %d\n", image.channels());

	//UBYTE THRESHOLD = 100;

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
#ifdef GPU_VER //normally ifndef, but needed to make this not run
	//threshold(THRESHOLD, image);
	
	//set up the kernel for the box filter
	UBYTE boxKernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	int kw = 3;
	int kh = 3;

	//here is where we would set up src, but image is already set so we'll use that

	//next we would convert to greyscale, but we already did that too

	//make two more Mats for box filtering
	Mat dst = image;
	Mat tmp = image;

	//get ready to time the filter
	hpt.TimeSinceLastCall();
	//apply the filter
	boxFilter(image.data, dst.data, image.cols, image.rows, boxKernel, kw, kh, tmp.data);
	//time the filter
	double boxTime = hpt.TimeSinceLastCall();
	printf("The box filter took %f seconds\n", boxTime);

	//show the image
	imshow(windowName, dst);
#endif

	//namedWindow("Display window", WINDOW_NORMAL);
	//imshow("Display window", image);

	waitKey(0);
	printf("\n\nFinal stats:\n");
	printf("Image size: %d x %d\n", image.cols, image.rows);
	printf("Kernel size: %d x %d\n", kw, kh);
	printf("Average time: %f\n", totalTime / double(runs));

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
void threshold(UBYTE threshold, Mat& image)
{
	UBYTE* end_data = image.data + (image.rows * image.cols);
	for (UBYTE* p = image.data; p < end_data; p++) {
		if (*p > threshold) {
			*p = 255;
		}
		else {
			*p = 0;
		}
	}
}

#ifdef GPU_VER
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
cudaError_t GPUThreshold(UBYTE threshold)
{
	//UBYTE* dev_threshold = 0;
	cudaError_t cudaStatus;
	dataSize = image.rows * image.cols * sizeof(UBYTE);
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

		/*cudaStatus = cudaMemcpy(dev_threshold, &threshold, sizeof(UBYTE), cudaMemcpyHostToDevice);
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
#endif

/*
 *	apply box filter on trackbar movement
 */
void boxTrackbar(int, void*)
{
	//record that we're running this again
	runs++;
	
	//here we would set up the kernel if it wasn't global

	//here is where we would set up src, but image is already set so we'll use that

	//next we would convert to greyscale, but we already did that too

	//make two more Mats for box filtering
	Mat dst, tmp;
	image.copyTo(dst);
	image.copyTo(tmp);

	//int kernel2[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	//get ready to time the filter
	hpt.TimeSinceLastCall();
	//apply the filter
	boxFilter(image.data, dst.data, image.cols, image.rows, boxKernel, kw, kh, tmp.data);
	//time the filter
	double boxTime = hpt.TimeSinceLastCall();
	totalTime += boxTime;
	//boxFilter(image.data, tmp.data, image.cols, image.rows, kernel2, kw, kh, dst.data);

	//print the results
	printf("Image size: %d x %d\n", image.cols, image.rows);
	printf("Kernel size: %d x %d\n", kw, kh);
	printf("Time this run: %f\n", boxTime);
	printf("Average time so far: %f\n", totalTime / double(runs));

	//show the image
	imshow(windowName, dst);
}

/*
 *	CPU box filter
 */
void boxFilter(UBYTE* src, UBYTE* dst, int w, int h, int* kernel, int kw, int kh, UBYTE* tmp)
{
	//used to easily move to surrounding pixels
	//int indices[] = {-(w+1), -w, -(w-1), -1, 0, 1, w-1, w, w+1};

	//leave an outer border so we don't need to handle literal edge cases
	int wEdge = kw / 2;
	int hEdge = kh / 2;
	//int hEdge = kh / 2;
	//printf("edge: %d\n", edge);

	//we divide each pixel's post-multiplication value by the sum of every kernel value, so may as well make it a variable
	int kernelSum = 0;
	for (int k = 0; k < kw*kh; k++) {
		kernelSum += kernel[k];
	}

	//go through each row (except the top and bottom)
	for (int i = hEdge; i < h - hEdge; i++) 
	{
		//go through each column within a row (except the left and right edges)
		for (int j = wEdge; j < w - wEdge; j++) 
		{
			//the current pixel's new value
			float current = 0.0f;
			//the current pixel's 1d array position
			int position = (i * w) + j;
			//go through each item in the kernel
			for (int kr = -hEdge; kr <= hEdge; kr++) 
			{
				for (int kc = -wEdge; kc <= wEdge; kc++) 
				{
					//printf("(%d, %d)\n", kr, kc);
					int relativePos = kr * w + kc;
					int kernelPos = (kr + hEdge) * kw + kc + wEdge;
					//printf("kr: %d, kc: %d, relativePos: %d, kernelPos: %d\n", kr, kc, relativePos, kernelPos);
					current += float(src[position + relativePos]) * float(kernel[kernelPos]);
				}
			}
			/*for (int k = 0; k < kw * kh; k++) 
			{
				//sum up the result of kernel value * original value for each neighboring pixel
				current += float(src[position + indices[k]]) * float(kernel[k]);
			}*/
			//if kernelSum isn't zero (dividing by zero considered harmful)
			if (kernelSum != 0) {
				//divide it by the sum of all the kernel values
				dst[position] = int(current / float(kernelSum));
			}
			else {
				dst[position] = int(current / 1.0f);
			}
		}
	}
}