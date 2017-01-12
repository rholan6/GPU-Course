
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../high_performance_timer/highperformancetimer.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>

using namespace std;

bool setupArrays(int** a, int** b, int** c, int size);
void fillArrays(int* a, int* b, int size);
cudaError_t copyArrays(int* a, int* b, int* c, int* dev_a, int* dev_b, int* dev_c);
cudaError_t setupGPU(int* dev_a, int* dev_b, int* dev_c, size_t size, cudaDeviceProp* dev_properties);
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
void addWithCPU(int*a, int*b, int*c, int size);
void cleanupCPU(int* a, int* b, int* c);
void cleanupGPU(int* dev_a, int* dev_b, int* dev_c);

/*
 *	addKernel
 */
__global__ void addKernel(int *c, const int *a, const int *b, const int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
}

/*
 *	main
 */
int main(int argc, char* argv[])
{
	//seed the rng for filling arrays later
	srand(time(NULL));

	//set up a variable to store cudaError_t from cuda functions
	cudaError_t cudaStatus;

	//ready the timer
	HighPrecisionTime hpt;
	//also some timer variables
	double totalCPUTime = 0.0;
	double averageCPUTime = 0.0;
	double totalGPUTime = 0.0;

	//make pointers for the arrays (we can malloc them later when we have a size for them)
	int *a, *b, *c;
	a = b = c = nullptr;

	//set defaults for array size and number of tests
	int arraySize = 1000;
	int repetitions = 100;
	//if the user provided values for either of the above, use those instead of the defaults
	if (argc > 1) {
		arraySize = stoi(argv[1]);
	}
	if (argc > 2) {
		repetitions = stoi(argv[2]);
	}
	//output the values being used
	cout << "Running " << repetitions << " tests on " << arraySize << "-element arrays" << endl;

	//exit code (set to 1 in cases of error)
	int retVal = 0;

	try
	{
		/*
			Allocate arrays
		*/
		//let the user know what's going on
		cout << "Allocating space for arrays" << endl;
		//since we know the size for these arrays, malloc a, b, and c
		if (!setupArrays(&a, &b, &c, arraySize)) {
			//if we can't malloc them, the show can't go on. just give up.
			throw "failed to allocate arrays";
		}

		/*
			Set up GPU
		*/
		int *dev_a, *dev_b, *dev_c;
		dev_a = dev_b = dev_c = nullptr;
		cudaStatus = setupGPU(/*fill me*/);

		/*
			Add with CPU
		*/
		cout << "Adding with the CPU" << endl;
		for (int i = 0; i < repetitions; i++) {
			//may as well start with new random values each time
			fillArrays(a, b, arraySize);
			//start the timer, add the arrays, and add the time elapsed to our total
			hpt.TimeSinceLastCall();
			addWithCPU(a, b, c, arraySize);
			totalCPUTime += hpt.TimeSinceLastCall();
		}
		cout << "Adding together " << arraySize << "-element arrays " << repetitions << " times with the CPU took " << totalCPUTime << " seconds" << endl;
		//figure out the average time for CPU adding
		averageCPUTime = totalCPUTime / repetitions;
		cout << "The average time for adding them with the CPU was " << averageCPUTime << " seconds\n" << endl;

		/*
			Add with GPU
		*/
		cout << "Adding with the GPU" << endl;
		for (int i = 0; i < repetitions; i++) {
			cudaStatus = addWithCuda(c, a, b, arraySize);
			if (cudaStatus != cudaSuccess) {
				//throw an exception if things went south
				throw("addWithCuda failed! on repetition " + i);
			}
		}

		/*
			Clean up
		*/

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			//throw an exception if things go south
			throw "cudaDeviceReset failed!";
		}
	}
	catch (char* e)
	{
		cerr << "ERROR: " << e << endl;
		retVal = 1;
	}

    return retVal;
}

/*
 *	setupArrays 
 */
//attempt to malloc a, b, and c
bool setupArrays(int** a, int** b, int** c, int size)
{
	//Assume things will work until they don't
	bool success = true;

	//try to find room for all the arrays
	try
	{
		//make some room in memory for these arrays
		*a = (int*)malloc(size * sizeof(int));
		if (*a == nullptr) {
			throw "malloc() FAILED on a";
		}

		*b = (int*)malloc(size * sizeof(int));
		if (*b == nullptr) {
			throw "malloc() FAILED on b";
		}

		*c = (int*)malloc(size * sizeof(int));
		if (*c == nullptr) {
			throw "malloc() FAILED on c";
		}
	}
	catch (char* e)
	{
		//print the error (ERROR: malloc() FAILED on [id])
		cerr << "ERROR: " << e << endl;
		//free anything that was successfully malloc'd
		cleanupCPU(*a, *b, *c);
		//we did not succeed
		success = false;
	}

	//tell the user how we did
	return success;
}

/*
 *	fillArrays
 */
//fills a and b with randomish numbers (c is getting overwritten anyway, so it doesn't really matter)
void fillArrays(int* a, int* b, int size)
{
	for (int i = 0; i < size; i++) {
		a[i] = rand() % 25;
		b[i] = rand() % 20;
	}
}

/*
 *	copyArrays
 */
//copy arrays to gpu memory
cudaError_t copyArrays(int* a, int* b, int* c, int* dev_a, int* dev_b, int* dev_c)
{
	//fill me
}

/*
 *	setupGPU
 */
//set up the GPU arrays and CUDA environment
cudaError_t setupGPU(int* dev_a, int* dev_b, int* dev_c, size_t size, cudaDeviceProp* dev_properties)
{
	cudaError_t cudaStatus;
	try
	{
		//select a GPU (0) and get its properties
		if (cudaSetDevice(0) != cudaSuccess) {
			throw "cudaSetDevice(0) failed";
		}
		if (cudaGetDeviceProperties(dev_properties, 0) != cudaSuccess) {
			throw "FAILED to get properties for device 0";
		}

		//find room for the three arrays
		cudaStatus = cudaMalloc(&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on dev_a";
		}

		cudaStatus = cudaMalloc(&dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on dev_b";
		}
		
		cudaStatus = cudaMalloc(&dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			throw "cudaMalloc failed on dev_c";
		}
	}
	catch (char* e) 
	{
		//catch the exception in more than name alone
		cerr << "Error: " << e << endl;
		//if we run into any problems, free anything that was successfully malloc'd
		cleanupGPU(dev_a, dev_b, dev_c);
	}

	//return cudaSuccess or whatever error we encountered
	return cudaStatus;
}


/*
 *	addWithCuda
 */
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	cudaDeviceProp dev_properties;

	try
	{
		cudaStatus = setupGPU(dev_a, dev_b, dev_c, size, &dev_properties);
		if (cudaStatus != cudaSuccess) {
			throw "Failed to prepare GPU";
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on a!";
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on b!";
		}

		//math
		int blocksNeeded = (size + dev_properties.maxThreadsPerBlock - 1) / dev_properties.maxThreadsPerBlock;

		addKernel <<<blocksNeeded, dev_properties.maxThreadsPerBlock >>>(dev_c, dev_a, dev_b, size);

		// Launch a kernel on the GPU with one thread for each element.
		//addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			//Beautiful
			string error = "addKernel launch failed: " + (string)cudaGetErrorString(cudaStatus);
			throw error.c_str();
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			//Such pretty. Wow.
			string error = "cudaDeviceSynchronize returned error code " + cudaStatus + (string)" after launching addKernel!";
			throw error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw "cudaMemcpy failed on dev_c!";
		}
	}
	catch (char* e)
	{
		cerr << "Error: " << e << endl;
		cleanupGPU(dev_a, dev_b, dev_c);
	}

    return cudaStatus;
}


/*
 *	addWithCPU
 */
//like addWithCuda, but on the cpu instead of the gpu
void addWithCPU(int*a, int*b, int*c, int size)
{
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}


/*
 *	cleanupCPU
 */
//free any allocated arrays from normal memory
void cleanupCPU(int* a, int* b, int* c)
{
	if (a != nullptr) {
		free(a);
	}
	if (b != nullptr) {
		free(b);
	}
	if (c != nullptr) {
		free(c);
	}
}

/*
 *	cleanupGPU
 */
//tie up loose ends with GPU memory
void cleanupGPU(int* dev_a, int* dev_b, int* dev_c)
{
	//free up arrays
	if (dev_c != 0) {
		cudaFree(dev_c);
	}
	if (dev_a != 0) {
		cudaFree(dev_a);
	}
	if (dev_b != 0) {
		cudaFree(dev_b);
	}
}