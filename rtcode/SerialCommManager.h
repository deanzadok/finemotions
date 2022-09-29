#pragma once

#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <vector>
#include <chrono>
using namespace std::chrono;

class SerialCommManager {

public:
	//properties
	DCB dcb;
	HANDLE hCom;
	BOOL fSuccess;
	LPCTSTR pcCommPort; // The serial port of the arduino

	// system parameters
	std::vector<int> handState = { 0, 0, 0, 0, 0 };
	double handStateTime;
	
	//methods
	SerialCommManager(LPCTSTR commPort);
	void printCommState(DCB dcb);
	BOOL updateHandState(std::vector<float> predProbabilities);
	char* createCommMessage();
	BOOL sendMessage();
	void printHandState();
};

