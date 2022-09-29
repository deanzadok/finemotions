#include "ONNXModelManager.h"


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


ONNXModelManager::ONNXModelManager() {

    // model private parameters
    const wchar_t* convBlockFilePath = L"C:\\Users\\User\\Documents\\Github\\NeuralHandshakeRT\\models\\model_convonly.onnx";
    const wchar_t* recBlockFilePath = L"C:\\Users\\User\\Documents\\Github\\NeuralHandshakeRT\\models\\model_reconly.onnx";

    // define environment with cuda support
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "nhs-inference");
    sessionOptions = new Ort::SessionOptions();
    auto* status = OrtSessionOptionsAppendExecutionProvider_CUDA(*sessionOptions, 0);
    if (status) {
        std::cout << "Error adding cuda provider\n";
        exit(1);
    }
    std::cout << "Added cuda support.\n";

    // allocate memory 
    allocator = new Ort::AllocatorWithDefaultOptions();

    // load conv block model
    convSession = new Ort::Session(*env, convBlockFilePath, *sessionOptions);

    // get input tensor type and shape
    auto inputTensorInfo = convSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Conv Input Type: " << inputType << std::endl;
    convInputDims = inputTensorInfo.GetShape();
    std::cout << "Conv Input Shape : ";
    if (convInputDims.at(0) < 0) {
        convInputDims[0] = convInputDims[0] * -1;
    }
    for (int64_t idim : convInputDims)
        std::cout << idim << '.';
    std::cout << std::endl;

    // get output tensor type and shape
    auto outputTensorInfo = convSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Conv Output Type: " << outputType << std::endl;
    convOutputDims = outputTensorInfo.GetShape();
    std::cout << "Conv Output Shape : ";
    if (convOutputDims.at(0) < 0) {
        convOutputDims[0] = convOutputDims[0] * -1;
    }
    for (int64_t odim : convOutputDims)
        std::cout << odim << '.';
    std::cout << std::endl;

    // load rec block model
    recSession = new Ort::Session(*env, recBlockFilePath, *sessionOptions);

    // get input tensor type and shape
    auto recInputTensorInfo = recSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType recInputType = recInputTensorInfo.GetElementType();
    std::cout << "Rec Input Type: " << recInputType << std::endl;
    recInputDims = recInputTensorInfo.GetShape();
    std::cout << "Rec Input Shape : ";
    if (recInputDims.at(0) < 0) {
        recInputDims[0] = recInputDims[0] * -1;
    }
    for (int64_t idim : recInputDims)
        std::cout << idim << '.';
    std::cout << std::endl;

    // get output tensor type and shape
    auto recOutputTensorInfo = recSession->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType recOutputType = recOutputTensorInfo.GetElementType();
    std::cout << "Rec Output Type: " << recOutputType << std::endl;
    recOutputDims = recOutputTensorInfo.GetShape();
    std::cout << "Rec Output Shape : ";
    if (recOutputDims.at(0) < 0) {
        recOutputDims[0] = recOutputDims[0] * -1;
    }
    if (recOutputDims.at(1) < 0) {
        recOutputDims[1] = recOutputDims[1] * -1;
    }
    if (recOutputDims.at(1) < sequenceSize) {
        recOutputDims[1] = recOutputDims[1] * sequenceSize;
    }
    for (int64_t odim : recOutputDims)
        std::cout << odim << '.';
    std::cout << std::endl;

    // get input and output node names for conv block
    const char* convInputName = convSession->GetInputName(0, *allocator);
    std::cout << "Conv Input Name: " << convInputName << std::endl;
    const char* convOutputName = convSession->GetOutputName(0, *allocator);
    std::cout << "Conv Output Name: " << convOutputName << std::endl;
    convInputNodeNames = { convInputName };
    convOutputNodeNames = { convOutputName };

    // get input and output node names for rec block
    const char* recInputName = recSession->GetInputName(0, *allocator);
    std::cout << "Rec Input Name: " << recInputName << std::endl;
    const char* recOutputName = recSession->GetOutputName(1, *allocator);
    std::cout << "Rec Output Name: " << recOutputName << std::endl;
    recInputNodeNames = { recInputName };
    recOutputNodeNames = { recOutputName };

    // fill rec block input
    for (int i = 0; i < sequenceSize; i++) {
        std::vector<float> featuresVector(convOutputDims[2], 0.0);
        recInputTensorValues.push_back(featuresVector);
    }

    // initial probabilities
    predictedProbabilities = { 0.0, 0.0, 0.0, 0.0, 0.0 };
}


float ONNXModelManager::approximateSigmoid(float x) {
    return 0.5 * (x / (1 + abs(x)) + 1);
}


void ONNXModelManager::computeConvInference(cv::Mat img) {

    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

    // prepare input and output tensors for conv block
    size_t convInputTensorSize = convInputDims[1] * convInputDims[2];
    std::vector<float> convInputTensorValues(convInputTensorSize);
    convInputTensorValues.assign(img.begin<float>(), img.end<float>());
    size_t convOutputTensorSize = convOutputDims[2];
    std::vector<float> convOutputTensorValues(convOutputTensorSize);

    // execute conv block inference
    std::vector<Ort::Value> convInputTensors;
    std::vector<Ort::Value> convOutputTensors;
    convInputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, convInputTensorValues.data(), img.total(), convInputDims.data(), convInputDims.size()));
    convOutputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, convOutputTensorValues.data(), convOutputDims[2], convOutputDims.data(), convOutputDims.size()));
    convSession->Run(Ort::RunOptions{ nullptr },
        convInputNodeNames.data(),   //
        convInputTensors.data(),      // input tensors
        1,      // 1
        convOutputNodeNames.data(),  //
        convOutputTensors.data(), 1); // 1

    // add conv output to rec input vector and make sure to keep last 8 valuess
    recInputTensorValues.push_back(convOutputTensorValues);
    recInputTensorValues.erase(recInputTensorValues.begin());
}


std::vector<float> ONNXModelManager::computeRecInference() {

    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

    // prepare input and output tensors for rec block
    size_t recInputTensorSize = recInputDims[1] * recInputDims[2];
    size_t recOutputTensorSize = recOutputDims[1] * recOutputDims[2];
    std::vector<float> recOutputTensorValues(recOutputTensorSize);

    // prepare rec block input
    std::vector<Ort::Value> recInputTensors;
    std::vector<Ort::Value> recOutputTensors;
    std::vector<float> recInputTensorValuesFlattened;
    for (auto const& v : recInputTensorValues) {
        recInputTensorValuesFlattened.insert(recInputTensorValuesFlattened.end(), v.begin(), v.end());
    }

    // execute rec block inference
    recInputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, recInputTensorValuesFlattened.data(), 4608 * 8, recInputDims.data(), recInputDims.size()));
    recOutputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, recOutputTensorValues.data(), 40, recOutputDims.data(), recOutputDims.size()));
    recSession->Run(Ort::RunOptions{ nullptr },
        recInputNodeNames.data(),   //
        recInputTensors.data(),      // input tensors
        1,      // 1
        recOutputNodeNames.data(),  //
        recOutputTensors.data(), 1); // 1

    // return last 5 probabilities
    for (int i = ((recInputDims[1] - 1) * numProbabilities); i < (recInputDims[1] * numProbabilities); i++) {
        //predictedProbabilities.push_back(recOutputTensorValues[i]); // logits only
        predictedProbabilities[i - 35] = approximateSigmoid(recOutputTensorValues[i]);
    }
    return predictedProbabilities;
}


std::vector<float> ONNXModelManager::computeFullInference(cv::Mat img) {

    double timestamp = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;

    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

    // prepare input and output tensors for conv block
    size_t convInputTensorSize = convInputDims[1] * convInputDims[2];
    std::vector<float> convInputTensorValues(convInputTensorSize);
    convInputTensorValues.assign(img.begin<float>(), img.end<float>());
    size_t convOutputTensorSize = convOutputDims[2];
    std::vector<float> convOutputTensorValues(convOutputTensorSize);

    // prepare input and output tensors for rec block
    size_t recInputTensorSize = recInputDims[1] * recInputDims[2];
    size_t recOutputTensorSize = recOutputDims[1] * recOutputDims[2];
    std::vector<float> recOutputTensorValues(recOutputTensorSize);

    // execute conv block inference
    std::vector<Ort::Value> convInputTensors;
    std::vector<Ort::Value> convOutputTensors;
    convInputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, convInputTensorValues.data(), img.total(), convInputDims.data(), convInputDims.size()));
    convOutputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, convOutputTensorValues.data(), convOutputDims[2], convOutputDims.data(), convOutputDims.size()));
    convSession->Run(Ort::RunOptions{ nullptr },
        convInputNodeNames.data(),   //
        convInputTensors.data(),      // input tensors
        1,      // 1
        convOutputNodeNames.data(),  //
        convOutputTensors.data(), 1); // 1

    // add conv output to rec input vector and make sure to keep last 8 valuess
    recInputTensorValues.push_back(convOutputTensorValues);
    recInputTensorValues.erase(recInputTensorValues.begin());

    // prepare rec block input
    std::vector<Ort::Value> recInputTensors;
    std::vector<Ort::Value> recOutputTensors;
    std::vector<float> recInputTensorValuesFlattened;
    for (auto const& v : recInputTensorValues) {
        recInputTensorValuesFlattened.insert(recInputTensorValuesFlattened.end(), v.begin(), v.end());
    }

    // execute rec block inference
    recInputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, recInputTensorValuesFlattened.data(), 4608 * 8, recInputDims.data(), recInputDims.size()));
    recOutputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, recOutputTensorValues.data(), 40, recOutputDims.data(), recOutputDims.size()));
    recSession->Run(Ort::RunOptions{ nullptr },
        recInputNodeNames.data(),   //
        recInputTensors.data(),      // input tensors
        1,      // 1
        recOutputNodeNames.data(),  //
        recOutputTensors.data(), 1); // 1

    // return last 5 probabilities
    std::vector<float> predictedProbabilities;
    for (int i = ((recInputDims[1] - 1) * numProbabilities); i < (recInputDims[1] * numProbabilities); i++) {
        //predictedProbabilities.push_back(recOutputTensorValues[i]); // logits only
        predictedProbabilities.push_back(approximateSigmoid(recOutputTensorValues[i]));
    }
    return predictedProbabilities;
}


void ONNXModelManager::printPredictions(std::vector<float> predictions) {

    std::cout << "[P1: " << predictions.at(0);
    for (int i = 1; i < numProbabilities; i++) {
        std::cout << ", P" << i+1 << ": " << std::fixed << std::setprecision(2) << predictions.at(i);
    }
    std::cout << "]" << std::endl;
}