// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SerialCommManager.h"
#include "ONNXModelManager.h"
#include "cast/cast.h"

#include <atomic>
#include <thread>

#define PRINT           std::cout << std::endl
#define PRINTSL         std::cout << "\r"
#define ERROR           std::cerr << std::endl
#define FAILURE         (-1)
#define SUCCESS         (0)

using namespace std::chrono;

static char* buffer_ = nullptr;
static int counter_ = 0;

// initiate FPS properties
double sum_time = 0;
int counter = 0;

// atomic variable for streaming loop
// when set to true, stop the program
std::atomic<bool> stop(false);

ONNXModelManager* onnxModelManager;
SerialCommManager* serialCommManager;

/******************************************
*            Clarius Callbacks
*******************************************/

/// callback for error messages
/// @param[in] err the error message sent from the casting module
void errorFn(const char* err)
{
    ERROR << "error: " << err;
}

/// callback for freeze state change
/// @param[in] val the freeze state value, 1 = frozen, 0 = imaging
void freezeFn(int val)
{
    PRINT << (val ? "frozen" : "imaging");
    counter_ = 0;
}

/// callback for button press
/// @param[in] btn the button that was pressed, 0 = up, 1 = down
/// @param[in] clicks # of clicks used
void buttonFn(int btn, int clicks)
{
    PRINT << (btn ? "down" : "up") << " button pressed, clicks: " << clicks;
}

/// callback for readback progreess
/// @param[in] progress the readback progress
void progressFn(int progress)
{
    PRINTSL << "downloading: " << progress << "%" << std::flush;
}

/// prints imu data
/// @param[in] npos the # of positional data points embedded with the frame
/// @param[in] pos the buffer of positional data
void printImuData(int npos, const CusPosInfo* pos)
{
    for (auto i = 0; i < npos; i++)
    {
        PRINT << "imu: " << i << ", time: " << pos[i].tm;
        PRINT << "accel: " << pos[i].ax << "," << pos[i].ay << "," << pos[i].az;
        PRINT << "gyro: " << pos[i].gx << "," << pos[i].gy << "," << pos[i].gz;
        PRINT << "magnet: " << pos[i].mx << "," << pos[i].my << "," << pos[i].mz;
    }
}

/// callback for a new image sent from the scanner
/// @param[in] newImage a pointer to the raw image bits of
/// @param[in] nfo the image properties
/// @param[in] npos the # of positional data points embedded with  the frame
/// @param[in] pos the buffer of positional data
void newProcessedImageFn(const void* newImage, const CusProcessedImageInfo* nfo, int npos, const CusPosInfo* pos)
{
    // initiate properties for recording loop   
    double timestamp, timestamp_end;

    timestamp = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;

    (void)newImage;
    (void)pos;

    // convert image data to OpenCV matrix
    cv::Mat mat = cv::Mat(nfo->height, nfo->width, CV_8UC4, (void*)newImage);
    ushort* data_start = mat.ptr<ushort>();
    size_t img_size = nfo->height * nfo->width * 4;
    memcpy(data_start, newImage, img_size);

    // convert image from uint8 [224,224,4] to float32 [224,224,1]
    mat.convertTo(mat, CV_32FC1);
    //cv::imwrite("C:\\Users\\User\\Documents\\Github\\Project\\mat.png", mat);

    // normalize image to the range of [0,1]
    mat = mat / 255.0;
    cv::dnn::blobFromImage(mat, mat);
    
    // compute conv block features only
    onnxModelManager->computeConvInference(mat);

    timestamp_end = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;

    // approximate fps
    /*counter++;
    if (counter > 20) {
        sum_time += (timestamp_end - timestamp);
        std::cout << "FPS: " << std::setprecision(16) << (counter - 20) / sum_time << std::endl;
    }*/
}

void clariusStreamingLoop() {

    int ret = 0;

    // recording loop - will stop when the atomic variable will get true
    while (!stop) {

        // images callback will be executed in background
        if (ret < 0)
            ERROR << "ERROR" << std::endl;
    }
}


void onnxExecutionLoop() {

    // initiate properties for recording loop
    double timestamp, timestamp_end;
    double onnx_sum_time = 0;
    int onnx_counter = 0;

    // recording loop - will stop when the atomic variable will get true
    while (!stop) {
        // get timestamp
        timestamp = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;

        // compute rec block inference only
        std::vector<float> predictions = onnxModelManager->computeRecInference();
        onnxModelManager->printPredictions(predictions);

        // measure fps
        timestamp_end = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;
        /*onnx_counter++;
        if (onnx_counter > 20) {
            onnx_sum_time += (timestamp_end - timestamp);
            std::cout << "FPS: " << std::setprecision(16) << (onnx_counter - 20) / onnx_sum_time << std::endl;
        }*/
    }
}


void serialCommExecutionLoop() {

    // initiate properties for recording loop
    double timestamp, timestamp_end;

    // recording loop - will stop when the atomic variable will get true
    while (!stop) {
        // get timestamp
        timestamp = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;

        // send instructions to the robotic hand based on model predictions
        BOOL res = serialCommManager->updateHandState(onnxModelManager->predictedProbabilities);
        //serialCommManager->printHandState();
        if (!res) {
            std::cout << "WARNING! Sending message failed." << std::endl;
        }

        // measure fps
        timestamp_end = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;
    }
}


int main(int argc, char* argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0) != 0 || setvbuf(stderr, nullptr, _IONBF, 0);
    
    // load and manage deap learning model using ONNX Runtime
    onnxModelManager = new ONNXModelManager();

    // initiate serial communication with the ESP module that will send instructions to the robotic hand
    serialCommManager = new SerialCommManager(TEXT("COM7"));

    // search for the clarius ultrasound device    
    bool ultrasoundDeviceAvailable = false;
    const int width = 224;
    const int height = 224;
    std::string keydir = "C:\\Users\\User\\Documents\\Github\\ViconCapture\\cast_key";
    std::string ipAddr = "192.168.1.1";
    unsigned int port = 5828;
    if (cusCastInit(argc, argv, keydir.c_str(), newProcessedImageFn, nullptr, nullptr, freezeFn, buttonFn, progressFn, errorFn, width, height) >= 0) {

        std::cout << "Clarius cast Initialized" << std::endl;
        if (cusCastConnect(ipAddr.c_str(), port, [](int ret) {

            if (ret == FAILURE)
                ERROR << "could not connect to scanner" << std::endl;
            else
                PRINT << "...connected, streaming port: " << ret << std::endl;

            }) < 0) {

            ERROR << "connection attempt failed" << std::endl;

        }
        else {

            // report connection
            ultrasoundDeviceAvailable = true;
            //PRINT << "...connected, streaming port: " << clariusGetUdpPort();
        }
    }
    else {

        ERROR << "Could not initialize Clarius caster" << std::endl;
    }

    // if ultrasound wasn't found - exit program
    if (ultrasoundDeviceAvailable == false) {
        exit(1);
    }

    // create threads
    std::thread ultrasoundLoopThread = std::thread(clariusStreamingLoop);
    std::thread onnxLoopThread = std::thread(onnxExecutionLoop);
    std::thread serialCommLoopThread = std::thread(serialCommExecutionLoop);

    // wait for key pressing and call to stop all threads
    std::cout << "Press the Enter key to finish streaming..." << std::endl;
    using namespace std::this_thread;
    sleep_until(system_clock::now() + seconds(1));

#ifdef _WIN32 || _WIN64
    system("pause");
#else;
    system("read -n1");
#endif

    stop = true;

    // join streaming threads
    ultrasoundLoopThread.join();
    onnxLoopThread.join();
    serialCommLoopThread.join();
    std::cout << "End of streaming." << std::endl;
}
