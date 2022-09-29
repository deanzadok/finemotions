/*
  Explanation
*/

/***********************************************************/
/*                Libraries                                */
/***********************************************************/
#include <BLEDevice.h>
#include <BLEAdvertisedDevice.h>

/*************************************************************************************/
/*                                  BlE Parameters                                   */
/*************************************************************************************/
#define MY_DEVICE_ADDRESS  "24:62:ab:f2:af:46" // add here you devicec address

#define HAND_DIRECT_EXECUTE_SERVICE_UUID     "e0198000-7544-42c1-0000-b24344b6aa70"
#define EXECUTE_ON_WRITE_CHARACTERISTIC_UUID "e0198000-7544-42c1-0001-b24344b6aa70"

#define HAND_TRIGGER_SERVICE_UUID            "e0198002-7544-42c1-0000-b24344b6aa70"
#define TRIGGER_ON_WRITE_CHARACTERISTIC_UUID "e0198002-7544-42c1-0001-b24344b6aa70"

static BLEUUID serviceExeUUID(HAND_DIRECT_EXECUTE_SERVICE_UUID);// The remote service we wish to connect to.
static BLEUUID    charExeUUID(EXECUTE_ON_WRITE_CHARACTERISTIC_UUID);// The characteristic of the remote service we are interested in - execute.

static BLEUUID serviceTrigUUID(HAND_TRIGGER_SERVICE_UUID);// The remote service we wish to connect to.
static BLEUUID    charTrigUUID(TRIGGER_ON_WRITE_CHARACTERISTIC_UUID);// The characteristic of the remote service we are interested in - trigger

// Connection parameters:
static boolean doConnect = false;
static boolean isconnected = false;
static boolean doScan = true;
static BLERemoteCharacteristic* pRemoteCharExecute;
static BLERemoteCharacteristic* pRemoteCharTrigger;
static BLEAdvertisedDevice* myDevice;

unsigned char* msg;
uint8_t preset_id;

/*************************************************************************************/
/*                 Go to Sleep and Keep Scanning Setting:                            */
/*************************************************************************************/
#define DT_SCAN    5000 //if not connected to BLE device scan every 5 seconds.
#define DT_SLEEP    (10 * 60 * 1000) //if not connected to BLE device go to sleep after 10 minute.
unsigned long t_scan; //the elapsed time between BLE scanning
unsigned long t_disconnected; //the elapsed time from a disconecting event

/*************************************************************************************/
/*                              Your System Setting:                                 */
/*************************************************************************************/
#define SwitchLeg_PRESET_TRIGGER 0
#define CLICK_TIME        1000   // the max time for a single click
#define NOISE_TIME        50     // the min time for a single click
#define BTWN_CLICKS_TIME  450  //the max time between two successive clicks
#define CLICKS_NUM        2     // the clicks counter variable
//switch parameters:
const int buttonPin = 4; // the pin which is connected to the button
bool button_state = LOW; // the current state of the button
bool button_last = LOW; // the last state of the button
unsigned long StartTime = millis(); // the time when a click was started
unsigned long EndTime = millis();  //the time when a single click was ended

int NumClick = 0; // the clicks counter variable
int pre_NumClick = 0; // the previous clicks counter variable

//unsigned char close_task[] =  {5, 0b11111000, 20, 0b01111000, 0b11111000}; //movement length byte, torque,time,active motor, motor direction
unsigned char hand_task[] =  {5, 0b11111000, 20, 0b01111000, 0b00000000}; //movement length byte, torque,time,active motor, motor direction
unsigned char hand_state = 0b00000000;

const unsigned int MAX_MESSAGE_LENGTH = 12;

/*************************************************************************************/
/*                  BLE Class Definition and Set Callbacks:                          */
/*************************************************************************************/
class MyClientCallback : public BLEClientCallbacks {
  void onConnect(BLEClient* pclient) {
  }

  void onDisconnect(BLEClient* pclient) {
    
    isconnected = false;
    Serial.println("onDisconnect");
    doScan = true;
    t_disconnected = millis();
  }
};

bool connectToServer() {
  
    Serial.print("Forming a connection to ");
    Serial.println(myDevice->getAddress().toString().c_str());
    
    BLEClient*  pClient  = BLEDevice::createClient();
    Serial.println(" - Created client");

    pClient->setClientCallbacks(new MyClientCallback());

    // Connect to the remove BLE Server.
    pClient->connect(myDevice);  // if you pass BLEAdvertisedDevice instead of address, it will be recognized type of peer device address (public or private)
    Serial.println(" - Connected to server");

    // Execute Charachteristics:
    // Obtain a reference to the service we are after in the remote BLE server.
    BLERemoteService* pRemoteExeService = pClient->getService(serviceExeUUID);
    if (pRemoteExeService == nullptr) {
      Serial.print("Failed to find our Execute service UUID: ");
      Serial.println(serviceExeUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Execute service");
    // Obtain a reference to the characteristic in the service of the remote BLE server.
    pRemoteCharExecute = pRemoteExeService->getCharacteristic(charExeUUID);
    if (pRemoteCharExecute == nullptr) {
      Serial.print("Failed to find our Execute characteristic UUID: ");
      Serial.println(charExeUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Execute characteristic");

    
    // Read the value of the characteristic.
    if(pRemoteCharExecute->canRead()) {
      std::string value = pRemoteCharExecute->readValue();
      Serial.print("Execute: The characteristic value was: ");
      Serial.println(value.c_str());
    }
    
    // Trigger Charachteristics:
    // Obtain a reference to the service we are after in the remote BLE server.
    BLERemoteService* pRemoteTrigService = pClient->getService(serviceTrigUUID);
    if (pRemoteTrigService == nullptr) {
      Serial.print("Failed to find our Trigger service UUID: ");
      Serial.println(serviceTrigUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Trigger service");
    // Obtain a reference to the characteristic in the service of the remote BLE server.
    pRemoteCharTrigger = pRemoteTrigService->getCharacteristic(charTrigUUID);
    if (pRemoteCharTrigger == nullptr) {
      Serial.print("Failed to find our Trigger characteristic UUID: ");
      Serial.println(charTrigUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Trigger characteristic");
 
    // Read the value of the characteristic.
    if(pRemoteCharTrigger->canRead()) {
      std::string value = pRemoteCharTrigger->readValue();
      Serial.print("Trigger: The characteristic value was: ");
      Serial.println(value.c_str());
    }

    isconnected = true;
    return isconnected;
}

// Scan for BLE servers and find the first one that advertises the service we are looking for.
class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {  // Called for each advertising BLE server.
  void onResult(BLEAdvertisedDevice advertisedDevice) {
    Serial.print("BLE Advertised Device found: ");
    Serial.println(advertisedDevice.toString().c_str());

    // We have found a device, let us now see if it contains the service we are looking for.
    if (((String)advertisedDevice.getAddress().toString().c_str()).equals(MY_DEVICE_ADDRESS)) {

      BLEDevice::getScan()->stop();
      myDevice = new BLEAdvertisedDevice(advertisedDevice);
      doConnect = true;
      doScan = false;

    } // Found our server
  } // onResult
}; // MyAdvertisedDeviceCallbacks

void InitBLE() {
  BLEDevice::init("SwitchLeg");
  // Retrieve a Scanner and set the callback we want to use to be informed when we
  // have detected a new device.  Specify that we want active scanning and start the
  // scan to run for 5 seconds.
  BLEScan* pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setInterval(1349);
  pBLEScan->setWindow(449);
  pBLEScan->setActiveScan(true);
  pBLEScan->start(1, false);
}

/******************************************************************************/
/*                          Setup Function:                                   */
/******************************************************************************/
void setup() {
  Serial.println("Device booted.");
  
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  // Begin communication with PC
  Serial.begin(57600);

  // Create the BLE Device
  InitBLE();
  t_scan = millis();

  Serial.println("BLE initiated.");

  // enable deep sleep mode for the esp32:
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4, 1); //1 = High, 0 = Low , same GPIO as the button pin
  t_disconnected = millis();

  // initate message
  msg = hand_task;
}

/******************************************************************************/
/*                           Loop Function:                                   */
/******************************************************************************/

// the loop function runs over and over again forever
void loop() {

  digitalWrite(LED_BUILTIN, LOW);

  if (doConnect == true) { //TRUE when we scanned and found the desired BLE server
    connectToServer();
    
    if (isconnected)
      Serial.println("We are now connected to the BLE Server."); // connect to the server. TRUE if connection was established
    else
      Serial.println("We have failed to connect to the server; there is nothin more we will do.");
    
    doConnect = false; //no need to reconnect to the server
  }

  // send intructions to the hand
  if (isconnected) {

    //msg[4] = 0b00000000;
    if (Serial.available() > 0) {
      char dataBuffer[2] = "\0";
      //unsigned char dataBuffer = 0b00000000;

      //Read the next available byte in the serial receive buffer
      int numReceived = Serial.readBytes(dataBuffer,1);//It require two things, variable name to read into, number of bytes to read.

      // send message only if instructions were modified
      if (msg[4] != dataBuffer[0]) {
        msg[4] = dataBuffer[0];
        pRemoteCharExecute->writeValue(msg,msg[0]);
        Serial.println("Message sent");
        digitalWrite(LED_BUILTIN, HIGH);
        delay(50);
      }
    }

         
  } else { //not connected
    //scanning for server:
    if((millis()-t_scan>DT_SCAN)){ //not connected
      //BLEDevice::getScan()->start(0);  // start to scan after disconnect. 0 - scan for infinity, 1-1 sec, ..
      Serial.println("Scanning...");
      BLEDevice::getScan()->start(1, false);
      t_scan = millis();
      
    } else if (millis()-t_disconnected > DT_SLEEP){ // going to sleep if long time without connection
      //Go to sleep now
      Serial.println("Going to sleep now");
      esp_deep_sleep_start();
    }
  }

  //delay(50); // bluetooth stack will go into congestion, if too many packets are sent  
}
