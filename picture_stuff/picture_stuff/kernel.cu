//since threshold has a gpu and cpu version, may as well have a way to not setup cuda if we only want the cpu version
#define GPU_VER

//cuda libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//printf
#include <stdio.h>

//opencv libraries
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//timer
#include "hpt.h"

using namespace cv;

//unsigned char gets used enough to justify a typedef
typedef unsigned char UBYTE;

/*
 *	Global Variables
 */
//threshold for... threshold
UBYTE thresholdSlider = 128;
//GPU pointers for image data
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
//one part of edge detection
//int boxKernel[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
//box kernel dimensions
const int kw = 69;
const int kh = 69;
//blur filter (filled with ones later)
UBYTE boxKernel[kw * kh];

#ifdef GPU_VER
/*
 *	Threshold Kernel
 */
__global__ void thresholdKernel(UBYTE* original, UBYTE* modified, UBYTE threshold, int size)
{
	//figure out what pixel we're working with
	int current = blockIdx.x * blockDim.x + threadIdx.x;
	//make sure we're in bounds
	if (current < size)
	{
		//make light pixels white and dark pixels black
		if (original[current] > threshold) {
			modified[current] = 255;
		}
		else {
			modified[current] = 0;
		}
	}
}

/*
 *	gpu constants
 */
__constant__ UBYTE dev_kernel[kw * kh];
__constant__ float gpu_kernelSum;

/*
 *	convolution kernel
 */
__global__ void convolutionKernel(UBYTE* src, int w, int h, UBYTE* dst, int edge)
{
	//figure out our current coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//the current pixel's 1d array position
	int position = (y * w) + x;

	//make sure we're in bounds
	if (position < (w*h))
	{
		//the current pixel's new value
		float current = 0.0f;
		//go through each item in the kernel
		for (int kr = -edge; kr <= edge; kr++)
		{
			for (int kc = -edge; kc <= edge; kc++)
			{
				//figure out how to reach the current relative position (up left, up, up right, etc.)
				int relativePos = kr * w + kc;
				//get the index for our current kernel position
				int kernelPos = (kr + edge) * kw + kc + edge;
				//multiply the current kernel position by the equivalent pixel, add the result to our running sum
				current += float(src[position + relativePos]) * float(dev_kernel[kernelPos]);
			}
		}
		//if kernelSum isn't zero (dividing by zero considered harmful)
		if (int(gpu_kernelSum) != 0) {
			//divide our sum for this pixel by the sum of all the kernel values, make that the new pixel value
			dst[position] = int(current / gpu_kernelSum);
		}
		else {
			//divide our running sum by one for giggles, make it the new pixel value
			dst[position] = int(current / 1.0f);
		}
	}
}
#endif

/*
 *	Function prototypes
 */
void threshold(UBYTE threshold, Mat& image);
#ifdef GPU_VER
void on_trackbar(int, void*);
cudaError_t GPUThreshold(UBYTE threshold);
cudaError_t GPUConvolution(UBYTE* dst);
#endif
void boxTrackbar(int, void*);
void boxFilter(UBYTE* src, UBYTE* dst, int w, int h, UBYTE* kernel, int kw, int kh, UBYTE* tmp);


/*
 *	Main
 */
int main(int argc, char* argv[])
{
	//Make sure we have an image to play with
	if (argc != 2) {
		printf("Usage: %s ImageToLoadAndDisplay\n", argv[0]);
		exit(1);
	}

	//Mat image;
	//Read in our image
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//printf("Number of channels: %d\n", image.channels());

	//Make sure our image was successfully read in
	if (!image.data) {
		printf("Could not find or open the image\n");
		exit(1);
	}

	//Convert image to greyscale, display it, and wait for input before going further
	cvtColor(image, image, COLOR_RGB2GRAY);
	namedWindow(windowName, WINDOW_NORMAL);
	imshow(windowName, image);
	waitKey(0);

	//fill kernel with ones
	for (int i = 0; i < kw * kh; i++) {
		boxKernel[i] = 1;
	}

	//set up a trackbar for quick timing
	//createTrackbar("Threshold", windowName, (int*)&thresholdSlider, 255, boxTrackbar);
	//boxTrackbar(thresholdSlider, 0);

	//printf("Number of channels: %d\n", image.channels());

	//UBYTE THRESHOLD = 100;

#ifdef GPU_VER
	//Get ready to catch errors
	cudaError_t cudaStatus;

	//Set cuda device to card 0 (since most cuda-capable setups have a cuda-capable card 0, while not all setups have any other cards)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to set CUDA device");
	}

	//Make a copy of our image to store the filtered image in
	Mat dst;
	image.copyTo(dst);

	//Pass this copy to a function that finishes setting up cuda and applies the box filter
	cudaStatus = GPUConvolution(dst.data);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to apply box filter\n");
		//If the filter failed, just give up. Clean up graphics memory and call it a day.
		goto cleanup;
	}

	//show the result and wait for a keypress before cleaning up and exiting
	imshow(windowName, dst);
	waitKey(0);
	goto cleanup;

	//call this once so I don't need to move all the cuda setup code (and to see a thresholded image from the start)
	/*cudaStatus = GPUThreshold(thresholdSlider);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to apply threshold filter\n");
		//not sure what state the image will be in after things fail, but it's probably better to just stop
		exit(1);
	}

	namedWindow(windowName, WINDOW_NORMAL);
	createTrackbar("Threshold", windowName, (int*)&thresholdSlider, 255, on_trackbar);
	on_trackbar(thresholdSlider, 0);
	imshow(windowName, image);*/
#endif
#ifndef GPU_VER
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

	//timing cpu box filter
	//waitKey(0);
	//printf("\n\nFinal stats:\n");
	//printf("Image size: %d x %d\n", image.cols, image.rows);
	//printf("Kernel size: %d x %d\n", kw, kh);
	//printf("Average time: %f\n", totalTime / double(runs));

#ifdef GPU_VER
cleanup:
	//free memory
	if (dev_original != 0) {
		cudaFree(dev_original);
	}
	if (dev_modified != 0) {
		cudaFree(dev_modified);
	}
	printf("GPU memory freed\n");

	//reset cuda device
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED on cudaDeviceReset\n");
		//if it doesn't wanna reset, not much we can do
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
	//Get a pointer to the end of our UBYTE(pixel) array
	UBYTE* end_data = image.data + (image.rows * image.cols);
	//Use pointers to go through our array (faster than using indexing)
	for (UBYTE* p = image.data; p < end_data; p++) {
		//Make light pixels white and dark pixels black
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
	//Get ready to catch cuda errors
	cudaError_t cudaStatus;
	//Print the threshold for debugging
	printf("Threshold: %d\n", thresholdSlider);
	
	//Determine how many blocks of 1024 threads it'll take to do this
	int numBlocks = (1023 + image.rows * image.cols) / 1024;
	//Fire up our kernel
	thresholdKernel <<<numBlocks, 1024>>> (dev_original, dev_modified, thresholdSlider, dataSize);
	
	//Wait for the kernel to finish and catch any errors from running it
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR with threshold kernel: %d\n", cudaStatus);
	}
	//Only copy the image back if threshold worked
	else {
		//Copy the modified image to cpu memory
		cudaStatus = cudaMemcpy(image.data, dev_modified, dataSize, cudaMemcpyDeviceToHost);
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FAILED to copy modified data to CPU memory\n");
	}
	//only show the image if everything worked
	else {
		imshow(windowName, image);
	}
}

/*
 *	GPU Threshold
 */
cudaError_t GPUThreshold(UBYTE threshold)
{
	//UBYTE* dev_threshold = 0;
	//Get ready to catch cuda errors
	cudaError_t cudaStatus;
	//set dataSize to... data's size
	dataSize = image.rows * image.cols * sizeof(UBYTE);
	try
	{
		//Make some room for our original image's data
		cudaStatus = cudaMalloc((void**)&dev_original, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_original";
		}

		//make room for modified image data
		cudaStatus = cudaMalloc((void**)&dev_modified, dataSize);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_modified";
		}

		/*cudaStatus = cudaMalloc((void**)&dev_threshold, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to allocate dev_threshold";
		}*/

		//copy the image's data to the GPU
		cudaStatus = cudaMemcpy(dev_original, image.data, dataSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy image data to GPU memory";
		}

		/*cudaStatus = cudaMemcpy(dev_threshold, &threshold, sizeof(UBYTE), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy threshold to GPU memory";
		}*/
		
		//Figure out (and print) how many blocks of 1024 threads we need to handle the image
		printf("Setting numblocks\n");
		int numBlocks = (1023 + image.rows * image.cols) / 1024;
		printf("%d blocks\n", numBlocks);
		//Fire up the kernel
		thresholdKernel<<<numBlocks, 1024>>>(dev_original, dev_modified, threshold, dataSize);

		//Get any errors from starting the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw "FAILED launching thresholdKernel";
		}

		//Wait for the kernel to finish and check for errors running it
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw "ERROR after launching thresholdKernel";
		}

		//Copy the thresholded image over to cpu memory
		cudaStatus = cudaMemcpy(image.data, dev_modified, dataSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy modified data to CPU memory";
		}
	}
	catch (char* e) {
		//Print the error description (thrown and caught as an exception) and print the cuda error code and string
		fprintf(stderr, "%s: %d: %s\n", e, cudaStatus, cudaGetErrorString(cudaStatus));
		//free memory
		if (dev_original != 0) {
			cudaFree(dev_original);
		}
		if (dev_modified != 0) {
			cudaFree(dev_modified);
		}
		printf("GPU memory freed\n");
	}

	//Say whether we succeeded or failed
	return cudaStatus;
}

/*
	GPU Box filter, etc.
 */
cudaError_t GPUConvolution(UBYTE* dst)
{
	//For catching cuda errors
	cudaError_t cudaStatus;
	//I'm not sure why I don't just set this in main, but oh well
	dataSize = image.rows * image.cols;
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

		cudaStatus = cudaMemcpy(dev_original, image.data, dataSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy image data to GPU memory";
		}

		cudaStatus = cudaMemcpyToSymbol(dev_kernel, &boxKernel, kw * kh * sizeof(UBYTE), 0, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy box kernel to GPU memory";
		}

		//compute kernel sum
		float sum = 0.0f;
		for (int i = 0; i < kw * kh; i++) {
			sum += boxKernel[i];
		}

		cudaStatus = cudaMemcpyToSymbol(gpu_kernelSum, &sum, sizeof(float), 0, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy kernel sum to GPU memory";
		}

		printf("Setting numblocks\n");
		int numBlocks = (1023 + image.rows * image.cols) / 1024;
		printf("%d blocks\n", numBlocks);
		convolutionKernel<<<numBlocks, 1024 >>>(dev_original, image.cols, image.rows, dev_modified, kw / 2);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw "FAILED launching convolution kernel";
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw "ERROR after launching convolution kernel";
		}

		cudaStatus = cudaMemcpy(dst, dev_modified, dataSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "FAILED to copy modified data to CPU memory";
		}
	}
	catch (char* e)
	{
		fprintf(stderr, "ERROR: %s: %d: %s\n", e, cudaStatus, cudaGetErrorString(cudaStatus));
	}

	if (dev_original != 0) {
		cudaFree(dev_original);
	}
	if (dev_modified != 0) {
		cudaFree(dev_modified);
	}
	printf("GPU memory freed\n");

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
void boxFilter(UBYTE* src, UBYTE* dst, int w, int h, UBYTE* kernel, int kw, int kh, UBYTE* tmp)
{
	//leave an outer border so we don't need to handle literal edge cases
	int wEdge = kw / 2;
	int hEdge = kh / 2;

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