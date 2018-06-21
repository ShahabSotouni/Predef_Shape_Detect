// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
#include <fstream>

using namespace cv;
using namespace std;
//******* Global vars ****************

int iLowH = 90;
 int iHighH = 150;

 int iLowS = 40; 
 int iHighS = 255;

 int iLowV = 40;
 int iHighV = 255;

Size outsize =Size(720,480);

  ofstream myfile;

//************* Local Functions *******************
static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage to find squares in a list of images\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 5;
const char* wndname = "H Detection";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


//extract topography
static void extopo(const Mat& image, vector<vector<Point> >& qcontours ){
   // cout<<"\n******************\nNewFrame\n";
     Mat nimg(outsize.height,outsize.width,CV_8UC3,Scalar(20,0,0));
   for( int i = 0; i< qcontours.size(); i++ )
     {
         
         Moments mom = moments(qcontours[i]); 
         double hu[7];
         HuMoments(mom, hu); //  in hu are  7 Hu-Moments
          myfile<<contourArea(qcontours[i])<<",";
         for(int k=0;k<7;k++)
             myfile<<hu[k]<<",";
         
             myfile<<"\n";
         
         bool isOK=true;
             isOK=isOK && ( hu[0]> 0.39  && hu[0]<0.55  ) ;
             isOK=isOK && ( hu[1]> 0.025  && hu[1]<0.78  ) ;
             isOK=isOK && ( hu[2]> 0.0  && hu[2]<0.3  ) ;
             isOK=isOK && ( hu[3]> 0  && hu[3]<1.73e-3 ) ;
             isOK=isOK && ( hu[4]>-1.16e-6 && hu[4]<1.16e-6 ) ;
             isOK=isOK && ( hu[5]>-2.1e-4  && hu[5]<2.1e-4 ) ;
             isOK=isOK && ( hu[6]>-1.2e-5  && hu[6]<1.2e-5 ) ;
        
         if (isOK)
               drawContours( nimg, qcontours, i, Scalar(0,250,250), -1, 8, RETR_LIST, 0, Point() );//*/
         /*/
   RotatedRect MARect=minAreaRect(qcontours[i]);
   
       
     //    boundingRect ()
   Rect BdRect= boundingRect(qcontours[i]);
    float crrangle=MARect.angle; //corrected angle
         if (crrangle<-45) crrangle=90+crrangle;
   Mat RotMat2d= getRotationMatrix2D(MARect.center, crrangle,1);
               cout<<"Rotation Angle: "<<crrangle<<"\n";
         
    vector<Point> rtcontour;
    transform(qcontours[i], rtcontour, RotMat2d);
    vector<vector<Point> > rtcontourvect;
         rtcontourvect.push_back(rtcontour);
         
         Scalar color = Scalar( 200,0,0 );
          drawContours( image, rtcontourvect, 0, color, 2, 8, RETR_LIST, 0, Point() );//*/
    
     }
    
      imshow("extopo output", nimg); //show the extopo output

//waitKey(0);
    
}









static void find( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    
    int erosion_size=3;
    
    Mat element = getStructuringElement(  MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
     /// Apply the erosion/Dilatation operation
  erode( image, image, element );
 dilate( image, image, element );
	
	 dilate( image, image, element );
 erode( image, image, element );
	
  vector<vector<Point> > contours;
            findContours(image, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);


// Qualify and Draw Contours
    vector<vector<Point> > qcontours;
RNG rng(12345);
//Mat contourimg=image.clone();
Mat contourimg=Mat::zeros( image.size(), CV_8UC3 );
 vector<Point> approx;
 for( int i = 0; i< contours.size(); i++ )
     {
          if(       fabs(contourArea(contours[i])) > outsize.area()/200 )                {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	 drawContours( contourimg, contours, i, color, 2, 8, RETR_LIST, 0, Point() );
              qcontours.push_back(contours[i]);
       /*/     
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.1, false);//arcLength(Mat(contours[i]), true)*0.01
        const Point* p = &approx[0];
        int n = (int)approx.size();
        polylines(contourimg, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);//*/
}}
 if( !contourimg.empty() )
        {
	imshow( "Contours", contourimg );
	}
//****************/
extopo( image,qcontours );
           
                }
            


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p-> x > 3 && p->y > 3)
          polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, image);
}





static void createcontrol(){
 namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

 //Create trackbars in "Control" window
 cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
 cvCreateTrackbar("HighH", "Control", &iHighH, 179);

 cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
 cvCreateTrackbar("HighS", "Control", &iHighS, 255);

 cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
 cvCreateTrackbar("HighV", "Control", &iHighV, 255);


}

static void tresh(Mat& image,Mat& imgThresholded){


    Mat imgHSV;
    cvtColor(image, imgHSV, COLOR_BGR2HSV);
    GaussianBlur(imgHSV, imgHSV, Size(7,7), 1.5, 1.5);
	  

	
	  	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
	
	 	 //morphological opening (remove small objects from the foreground)
	 	 erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
 

	  	//morphological closing (fill small holes in the foreground)
	 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
   erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		  imshow("Thresholded Image", imgThresholded); //show the thresholded image



}


static void extrctwtpad(Mat& image){
	
	 Mat imgHSV;
	Mat imgThresholded;
    cvtColor(image, imgHSV, COLOR_BGR2HSV);
    GaussianBlur(imgHSV, imgHSV, Size(7,7), 1.5, 1.5);
	  

	
	  	inRange(imgHSV, Scalar(0, 0, 255/3), Scalar(180, 255/3, 255), imgThresholded); //Threshold the image
	
	 	 //morphological opening (remove small objects from the foreground)
	 	 erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
 

	  	//morphological closing (fill small holes in the foreground)
	 	 dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
   erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	//	  imshow("White Thresholded Image", imgThresholded); //show the thresholded image
	
	
	

  vector<vector<Point> > contours;
            findContours(imgThresholded, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

// Qualify Contour and Create Mask
    vector<vector<Point> > qcontours;
int bigindex=0;
 for( int i = 0; i< contours.size(); i++ )
       	if(contourArea(contours[i])>contourArea(contours[bigindex]))
			bigindex=i;
		   
    Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);   
	drawContours(mask, contours, bigindex, Scalar(255), CV_FILLED);
	//imshow("Mask", mask); 
	Mat masked=Mat::zeros( image.size(), CV_8UC3 );
	image.copyTo(masked, mask);
	masked.copyTo(image);
	

	
	
	
}




//************************

int main(int /*argc*/, char** /*argv*/)
{
    
    help();
    createcontrol();
    
    namedWindow( wndname, 1 );
    
  
      myfile.open ("humoments.csv");
      myfile << "area,h1,h2,h3,h4,h5,h6,h7\n";    
    
    vector<vector<Point> > squares;
 string filename = "H.mp4";
    
VideoCapture  cap(filename); // open the File
      if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the file" << endl;
         return -1;
    }

   while(true)
    {
         Mat inimage ;
        Mat image ;
	Mat imgThresholded ;
if (!cap.read(inimage)) // get a new frame from file
        {
             cout << "Cannot read a frame from video stream" << endl;
            waitKey(0);
             break;  //if not success, break loop
        }
        if( inimage.empty() )
        {
            cout << "Couldn't load "<< endl;
            continue;
        }


resize(inimage, image, outsize);
		
//		imshow("Image Before White Extract", image); 
		extrctwtpad(image);
		imshow("Image After White Extract", image); 
		
tresh( image, imgThresholded);

        find(imgThresholded, squares);
     //   drawSquares(image, squares);
        //imwrite( "out", image );
         waitKey(0);
    }

    return 0;
}
