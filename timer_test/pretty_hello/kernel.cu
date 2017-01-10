
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t setup(int* dev_a, int* dev_b, int* dev_c, size_t size);
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
	//exit code (set to 1 in cases of error)
	int retVal = 0;

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
		//print errors to stderr instead of stdout (good practice for larger programs with less important output going to stdout which the user can pipe elsewhere)
		cerr << "addWithCuda failed!" << endl;
		//after printing the error, exit with code 1 (indicating something went wrong)
        retVal = 1;
		goto end_label;
    }

	//leaving this as a printf because the equivalent cout would look terrible
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
		cerr << "cudaDeviceReset failed!" << endl;
		//exit with code 1 (to indicate things went south)
        retVal = 1;
		goto end_label;
    }

end_label:
    return retVal;
}

//set up the arrays and CUDA environment
cudaError_t setup(int* dev_a, int* dev_b, int* dev_c, size_t size) 
{
	cudaError_t cudaStatus;
	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			//cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
			throw "cudaSetDevice failed";
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc(&dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			//cerr << "cudaMalloc failed on dev_c!" << endl;
			throw "cudaMalloc failed on dev_c";
		}

		cudaStatus = cudaMalloc(&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			//cerr << "cudaMalloc failed on dev_a!" << endl;
			throw "cudaMalloc failed on dev_a";
		}

		cudaStatus = cudaMalloc(&dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			//cerr << "cudaMalloc failed on dev_b!" << endl;
			throw "cudaMalloc failed on dev_b";
		}
	}
	catch (char* e) 
	{
		//catch the exception in more than name alone
		cerr << "Error: " << e << endl;
		//if we run into any problems, free anything that was successfully malloc'd
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

	//return cudaSuccess or whatever error we encountered
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

	try
	{
		cudaStatus = setup(dev_a, dev_b, dev_c, size);


		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed on a!" << endl;
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed on b!" << endl;
			goto Error;
		}

		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, size >> > (dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << endl;
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaMemcpy failed on dev_c!" << endl;
			goto Error;
		}
	}
	catch (char* e)
	{
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
	}

    return cudaStatus;
}
