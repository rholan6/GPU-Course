
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../high_performance_timer/highperformancetimer.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
//#include <omp.h>

using namespace std;

bool allocateMem(int** a, int** b, int** c, int size);
void fillArrays(int* a, int* b, int* c, int arrSize);
void addVector(int* a, int* b, int* c, int arrSize);
void cleanup(int* a, int* b, int* c);

int main(int argc, char* argv[]) 
{
	//seed the rng for filling a and b with random numbers later
	srand(time(NULL));
	//set up some variables for timing fillArrays() later
	double totalTime = 0.0;
	HighPrecisionTime hpt;
	//cout << argc << endl;
	//cout << argv[0] << endl;

	//If the user doesn't give a size, tell them they should do that. Then exit so they can do that
	if (argc == 1)
	{
		cerr << "Usage: \"cpuarrayaddr (arraySize)\"" << endl;
		cerr << "Please don't forget to include arraySize" << endl;
		exit(1);
	}

	//if we've made it here, there's a size. store it as an int
	int arraySize = stoi(argv[1]);
	cout << "arraySize set to " << arraySize << endl;

	//make the arrays
	int* a, *b, *c;
	a = b = c = nullptr;

	//set aside some array space
	bool success = allocateMem(&a, &b, &c, arraySize);

	//setup tells the user if anything goes horribly wrong, but let's give some good news if it goes well
	if (success) 
	{
		cout << "All arrays allocated successfully" << endl;
		
		//it's hard to get an accurate time since last call if there is no last call
		//hpt.TimeSinceLastCall();
		//fill the arrays, time the process
		for (int i = 0; i < 100; i++) {
			//We want to fill the arrays if we succeeded
			fillArrays(a, b, c, arraySize);
			//time the process
			//totalTime += hpt.TimeSinceLastCall();
			hpt.TimeSinceLastCall();
			addVector(a, b, c, arraySize);
			totalTime += hpt.TimeSinceLastCall();
		}
		//convert totalTime to the average time (since that's the end goal) and print it
		totalTime = totalTime / 100;
		cout << "The average time to add up arrays was " << totalTime << " seconds." << endl;

		//Make sure the arrays were filled
		/*for (int i = 0; i < arraySize; i++) {
			cout << "a[" << i << "] = " << a[i] << endl;
			cout << "b[" << i << "] = " << b[i] << endl;
			cout << "c[" << i << "] = " << c[i] << endl;
		}*/

		//add the arrays' contents
		//addVector(a, b, c, arraySize);

		//make sure this worked
		/*for (int i = 0; i < arraySize; i++) {
			cout << "c[" << i << "] = " << c[i] << endl;
		}*/
	}

	//Free them
	cleanup(a, b, c);

/*#ifdef _WIN32 || WIN64
	system("pause");
#endif*/
	return 0;
}

bool allocateMem(int** a, int** b, int** c, int size)
{
	//set the return value to true (and switch it to false later if anything fails)
	bool retVal = true;
	//allocate memory for a, b, c. tell the user if any mallocs fail
	*a = (int*)malloc(size * sizeof(int));
	if (*a == nullptr) {
		cerr << "malloc() FAILED on a" << endl;
		retVal = false;
	}

	*b = (int*)malloc(size * sizeof(int));
	if (*b == nullptr) {
		cerr << "malloc() FAILED on b" << endl;
		retVal = false;
	}

	*c = (int*)malloc(size * sizeof(int));
	if (*c == nullptr) {
		cerr << "malloc() FAILED on c" << endl;
		retVal = false;
	}

	//determine if malloc succeeded
	/*if (*a != nullptr && *b != nullptr && *c != nullptr) {
		retVal = true;
	}*/

	return retVal;
}

void fillArrays(int* a, int* b, int* c, int arrSize)
{
	//fill a and b with small random numbers, fill c with 0
	//we can use one loop because all arrays are the same size
//#pragma omp parallel for
	for (int i = 0; i < arrSize; i++)
	{
		//0-19 seems smallish enough
		a[i] = rand() % 20;
		b[i] = rand() % 20;
		c[i] = 0;
	}
}

void addVector(int* a, int* b, int* c, int arrSize)
{
	for (int i = 0; i < arrSize; i++) {
		c[i] = a[i] + b[i];
	}
}

void cleanup(int* a, int* b, int* c)
{
	//free them
	if (a != nullptr) {
		free(a);
		cout << "Freed a" << endl;
		a = nullptr;
	}
	if (b != nullptr) {
		free(b);
		cout << "Freed b" << endl;
		b = nullptr;
	}
	if (c != nullptr) {
		free(c);
		cout << "Freed c" << endl;
		c = nullptr;
	}
}