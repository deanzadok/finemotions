#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <limits>
#include <fstream>
#include <exception>
#include <cmath>

#pragma comment(lib, "onnxruntime.lib")

class ONNXModelManager
{
public:
	// ONNX properties
	Ort::Env* env;
	Ort::SessionOptions* sessionOptions;
	Ort::AllocatorWithDefaultOptions* allocator;

	// model sessions
	Ort::Session* convSession;
	Ort::Session* recSession;

	// model session placeholders
	std::vector<int64_t> convInputDims;
	std::vector<int64_t> convOutputDims;
	std::vector<int64_t> recInputDims;
	std::vector<int64_t> recOutputDims;
	std::vector<const char*> convInputNodeNames;
	std::vector<const char*> convOutputNodeNames;
	std::vector<const char*> recInputNodeNames;
	std::vector<const char*> recOutputNodeNames;

	// model parameters 
	int sequenceSize = 8;
	int numProbabilities = 5;

	// temporary memory for rec input
	std::vector<std::vector<float>> recInputTensorValues;
	std::vector<float> predictedProbabilities; // hold the last 5 probabilities

	// FPS measurement
	float sum_time = 0.0;
	int counter = 0;

	// methods
	ONNXModelManager();
	float approximateSigmoid(float x);
	void computeConvInference(cv::Mat img);
	std::vector<float> computeRecInference();
	std::vector<float> computeFullInference(cv::Mat img);
	void printPredictions(std::vector<float> predictions);
};

