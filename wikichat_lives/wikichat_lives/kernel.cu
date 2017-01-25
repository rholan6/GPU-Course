
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "highperformancetimer.h"

#include <stdio.h>
//#include <stdlib.h>

#include <fstream>
using namespace std;

int main()
{
	//define a gb
	int giga = 1 << 30;
	//allocate a billion bytes or something
	//char* wiki = (char*)malloc(giga);
	char* wiki = new char[giga]();
	//set up a bitmap to record hits
	char* bitmap = new char[giga / 8]();
	//set up a timer
	HighPrecisionTime hpt;
	double readTime;

	//open enwiki-latest-abstract.xml
	//FILE* wikiFile = fopen("C:/Users/educ/Documents/enwiki-latest-abstract.xml", "r");
	ifstream fin;
	fin.open("C:/Users/educ/Documents/enwiki-latest-abstract.xml");
	//make sure we opened the file
	if (!fin.is_open()) {
		printf("Failed to open file\n");
		return 1;
	}

	//read first gb into wiki and time it
	hpt.TimeSinceLastCall();
	fin.read(wiki, giga);
	readTime = hpt.TimeSinceLastCall();
	//make sure we read a full gig from the file
	if(fin.fail()) {
		printf("Failed to read %d bytes of wikipedia file\n", giga);
	}

	system("pause");

	//what gets mallocd must get freed
	//free(wiki);
	delete[] wiki;
	delete[] bitmap;
	fin.close();

	//print time
	printf("Reading %d bytes of the Wikipedia file took %f seconds\n", giga, readTime);
	system("pause");

	return 0;
}