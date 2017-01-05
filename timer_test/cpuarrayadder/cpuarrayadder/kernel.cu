
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>

using namespace std;

bool allocateMem(int** a, int** b, int** c, int size);
void fillArrays(int* a, int* b, int* c, int arrSize);
void cleanup(int* a, int* b, int* c);

int main(int argc, char* argv[]) 
{
	//seed the rng for filling a and b with random numbers later
	srand(time(NULL));
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
	if (success) {
		cout << "All arrays allocated successfully" << endl;
		
		//We want to fill the arrays if we succeeded
		fillArrays(a, b, c, arraySize);
		//Make sure the arrays were filled
		for (int i = 0; i < arraySize; i++) {
			cout << "a[" << i << "] = " << a[i] << endl;
			cout << "b[" << i << "] = " << b[i] << endl;
			cout << "c[" << i << "] = " << c[i] << endl;
		}
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
	for (int i = 0; i < arrSize; i++)
	{
		//0-19 seems smallish enough
		a[i] = rand() % 20;
		b[i] = rand() % 20;
		c[i] = 0;
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