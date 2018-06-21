#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
      if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the web cam" << endl;
         return -1;
    }
	

  namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

 int iLowH = 0;
 int iHighH = 179;

 int iLowS = 0; 
 int iHighS = 255;

 int iLowV = 0;
 int iHighV = 255;

 //Create trackbars in "Control" window
 cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
 cvCreateTrackbar("HighH", "Control", &iHighH, 179);

 cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
 cvCreateTrackbar("HighS", "Control", &iHighS, 255);

 cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
 cvCreateTrackbar("HighV", "Control", &iHighV, 255);




     Mat imgHSV;
   
    while(true)
    {
        Mat imgOriginal;
        if (!cap.read(imgOriginal)) // get a new frame from camera
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;  //if not success, break loop
        }
      
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
        GaussianBlur(imgHSV, imgHSV, Size(7,7), 1.5, 1.5);
  

	//****************************//
	/*int iLowH=45/2,iHighH=70/2; //Yellow Color
	int iLowS=0.6*255,iHighS=255; //Yellow Color
	int iLowV=0.6*255,iHighV=255; //Yellow Color
	 */
 	 Mat imgThresholded;

  	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
      
 	 //morphological opening (remove small objects from the foreground)
 	 erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

  	//morphological closing (fill small holes in the foreground)
 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
	  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	  imshow("Thresholded Image", imgThresholded); //show the thresholded image
 	 imshow("Original", imgOriginal); //show the original image



        waitKey(1);
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
