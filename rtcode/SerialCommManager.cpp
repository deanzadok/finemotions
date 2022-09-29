#include "SerialCommManager.h"


SerialCommManager::SerialCommManager(LPCTSTR commPort) {

	pcCommPort = commPort;

    //  Open a handle to the specified com port.
    hCom = CreateFile(pcCommPort,
           GENERIC_READ | GENERIC_WRITE,
           0,      //  must be opened with exclusive-access
           NULL,   //  default security attributes
           OPEN_EXISTING, //  must use OPEN_EXISTING
           0,      //  not overlapped I/O
           NULL); //  hTemplate must be NULL for comm devices

    if (hCom == INVALID_HANDLE_VALUE)
    {
        //  Handle the error.
        printf("CreateFile failed with error %d.\n", GetLastError());
        exit (1);
    }

    //  Initialize the DCB structure.
    SecureZeroMemory(&dcb, sizeof(DCB));
    dcb.DCBlength = sizeof(DCB);

    //  Build on the current configuration by first retrieving all current settings
    fSuccess = GetCommState(hCom, &dcb);

    if (!fSuccess) {
        //  Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        exit (1);
    }

    //  Fill in some DCB values and set the com state: 
    //  57,600 bps, 8 data bits, no parity, and 1 stop bit.
    dcb.BaudRate = CBR_57600;     //  baud rate
    dcb.ByteSize = 8;             //  data size, xmit and rcv
    dcb.Parity = NOPARITY;        //  parity bit
    dcb.StopBits = ONESTOPBIT;    //  stop bit

    fSuccess = SetCommState(hCom, &dcb);

    if (!fSuccess)
    {
        //  Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        exit (1);
    }

    //  Get the comm config again.
    fSuccess = GetCommState(hCom, &dcb);

    if (!fSuccess)
    {
        //  Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        exit (1);
    }

    // set hand state time
    handStateTime = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;

    printCommState(dcb);       //  Output to console
    std::cout << "Serial port " << pcCommPort << " successfully reconfigured." << std::endl;
}


void SerialCommManager::printCommState(DCB dcb) {
    //  Print some of the DCB structure values
    _tprintf(TEXT("\nBaudRate = %d, ByteSize = %d, Parity = %d, StopBits = %d\n"),
        dcb.BaudRate,
        dcb.ByteSize,
        dcb.Parity,
        dcb.StopBits);
}


BOOL SerialCommManager::updateHandState(std::vector<float> predProbabilities) {

    std::vector<int> givenState(5);
    // round probabilities into discrete values
    for (int i = 0; i < 5; i++) {
        givenState[i] = round(predProbabilities[i]);
    }

    // set time of state arrival
    double givenTime = static_cast<double>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()) * 1e-9;

    // send instructions to the hand if changed
    if (givenState != handState && (givenTime - handStateTime) > 0.2 ) {
        handStateTime = givenTime;
        handState = givenState;
        BOOL res = sendMessage();
        return res;
    }
    return true;
}


char* SerialCommManager::createCommMessage() {

    char instructingMessage[] = "0";
    int handStateBytes = 0x00;

    // flex finger F^5 (Little Finger)
    if (handState[4] == 1) {
        handStateBytes += 0x08; // 00001000
    }
    // flex finger F^4
    if (handState[3] == 1) {
        handStateBytes += 0x10; // 00010000
    }
    // flex finger F^3
    if (handState[2] == 1) {
        handStateBytes += 0x20; // 00100000
    }
    // flex finger F^2 (Index Finger)
    if (handState[1] == 1) {
        handStateBytes += 0x40; // 01000000
    }

    // return instructing signal
    instructingMessage[0] = handStateBytes;
    return instructingMessage;
}


BOOL SerialCommManager::sendMessage() {

    char* instructingMessage = createCommMessage();

    OVERLAPPED osWrite = { 0 };
    DWORD dwToWrite = 1;
    DWORD dwWritten;
    DWORD dwRes;
    BOOL fRes;

    // Create this write operation's OVERLAPPED structure's hEvent
    osWrite.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (osWrite.hEvent == NULL)
        return FALSE; // error creating overlapped event handle

    // Issue write.
    if (!WriteFile(hCom, instructingMessage, dwToWrite, &dwWritten, &osWrite)) {
        if (GetLastError() != ERROR_IO_PENDING) {
            // WriteFile failed, but isn't delayed. Report error and abort.
            fRes = FALSE;
        }
        else {
            // Write is pending.
            dwRes = WaitForSingleObject(osWrite.hEvent, INFINITE);

            switch (dwRes) {
                // OVERLAPPED structure's event has been signaled.
            case WAIT_OBJECT_0:
                if (!GetOverlappedResult(hCom, &osWrite, &dwWritten, FALSE))
                    fRes = FALSE;
                else
                    // Write operation completed successfully.
                    fRes = TRUE;
                break;
            default:
                // An error has occurred in WaitForSingleObject.
                // This usually indicates a problem with the
                // OVERLAPPED structure's event handle.
                fRes = FALSE;
                break;
            }
        }
    }
    else
        // WriteFile completed immediately.
        fRes = TRUE;
    CloseHandle(osWrite.hEvent);
    return fRes;
}


void SerialCommManager::printHandState() {

    std::cout << "[" << handState.at(0);
    for (int i = 1; i < 5; i++) {
        std::cout << ", " << handState.at(i);
    }
    std::cout << "]" << std::endl;
}