#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open default webcam (0 = first camera)
    VideoCapture cap(0);
    
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam" << endl;
        return -1;
    }
    
    // Get camera properties
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    
    cout << "Webcam opened successfully!" << endl;
    cout << "Resolution: " << frame_width << "x" << frame_height << endl;
    cout << "FPS: " << fps << endl;
    cout << "Press 'q' to quit, 's' to save frame" << endl;
    
    Mat frame;
    int frame_count = 0;
    
    while (true) {
        // Capture frame
        cap >> frame;
        
        // Check if frame is empty
        if (frame.empty()) {
            cerr << "Error: Blank frame grabbed" << endl;
            break;
        }
        
        frame_count++;
        
        // Add frame counter text
        putText(frame, "Frame: " + to_string(frame_count), 
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 
                Scalar(0, 255, 0), 2);
        
        // Display the frame
        imshow("Webcam Test", frame);
        
        // Wait for key press (1ms)
        char key = waitKey(1);
        
        if (key == 'q' || key == 'Q') {
            cout << "Quitting..." << endl;
            break;
        }
        else if (key == 's' || key == 'S') {
            string filename = "webcam_capture_" + to_string(frame_count) + ".jpg";
            imwrite(filename, frame);
            cout << "Saved: " << filename << endl;
        }
    }
    
    // Release camera and close windows
    cap.release();
    destroyAllWindows();
    
    cout << "Total frames captured: " << frame_count << endl;
    
    return 0;
}
