#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <FlyCapture2.h>
#include <string>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/aruco/charuco.hpp>
#include "boost/asio.hpp"

#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998
#define PORT "10090"
#define IP "192.168.200.192"

using namespace cv;
using namespace FlyCapture2;
using namespace std;
using namespace boost::asio;



void showFrames() {
	Camera camera;
	CameraInfo camInfo;
	camera.Connect(0);
	camera.GetCameraInfo(&camInfo);
	camera.StartCapture();
	Image rawImage;
	Image rgbImage;
	UMat gpuimage;
	char key = 0;
	int threshhold = 254;
	int onInt = 0;
	vector<vector<Point>> contourpoints;
	Point angle;
	namedWindow("frame", WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL);
	resizeWindow("frame", 600, 800);

	io_service io_service;
	ip::udp::resolver resolver(io_service);
	ip::udp::resolver::query query(IP, PORT);
	ip::udp::resolver::iterator iter = resolver.resolve(query);

	ip::udp::socket socket(io_service);

	auto remote = iter->endpoint();

	socket.open(boost::asio::ip::udp::v4());

	float intrinsic[3][3] = { 5.266478108199441e+02, 0, 5.090669900210755e+02, 0, 5.266442917480177e+02, 4.056045815230510e+02, 0, 0, 1 };
	float distortion[1][4] = { -0.3452, 0.1265, 0,0 };
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, intrinsic);
	Mat distortion_coeffs = Mat(1, 4, CV_32FC1, distortion);
	UMat newundistort;
	Mat original;
	Size imageSize = Size(1024,768);
	Mat R;
	Mat mapx = Mat(imageSize, CV_32FC1);
	Mat mapy = Mat(imageSize, CV_32FC1);
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, imageSize, CV_32FC1, mapx, mapy);
	auto saveimage = true;

	while (key != 'q') {
		long double e1 = getTickCount();
		camera.RetrieveBuffer(&rawImage);
		rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);

		 original = Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData());
		 
		


		 remap(original, newundistort, mapx, mapy, INTER_LINEAR);
		 cvtColor(newundistort, gpuimage, COLOR_BGR2GRAY);
		
		
		threshold(gpuimage, gpuimage, threshhold, 255, THRESH_BINARY);
		
		findContours(gpuimage, contourpoints, RETR_TREE, CHAIN_APPROX_SIMPLE);
		if (contourpoints.size() > 2) { //If all 3 markers are found
			vector<vector<Point>> savePoint1;
			
			for (int j = 0; j < 3; j++) {  //puts only the 3 biggest contours into final vector, this acts as a filter
				int index = 0;
				double area = contourArea(contourpoints[0], false);
				for (int i = 0; i < contourpoints.size(); i++) {
					if (contourArea(contourpoints[i]) > area) {
						area = contourArea(contourpoints[i]);
						index = i;
					}


				}
				savePoint1.push_back(contourpoints[index]);
				contourpoints.erase(contourpoints.begin() + index);

			}

			vector<Point> markerLocations;
			for (int i = 0; i < 3; i++) { //this gets and saves the point location for each contour by doing a bounded rectangle average
				Rect rectangle = boundingRect(savePoint1[i]);

				markerLocations.push_back(Point(rectangle.x + rectangle.width / 2,
					rectangle.y + rectangle.height / 2));
			}
			double AtoB = sqrt(pow((markerLocations[0].x - markerLocations[1].x), 2) + // this finds the distances between each of the 3 points
				pow((markerLocations[0].y - markerLocations[1].y), 2));
			double BtoC = sqrt(pow((markerLocations[1].x - markerLocations[2].x), 2) +
				pow((markerLocations[1].y - markerLocations[2].y), 2));
			double AtoC = sqrt(pow((markerLocations[0].x - markerLocations[2].x), 2) +
				pow((markerLocations[0].y - markerLocations[2].y), 2));

			Point A;
			Point B;
			Point C;

			if (AtoB < BtoC && AtoB < AtoC) { //this finds and assigns point A and B as the 2 closest points and C as the farthest
				A = markerLocations[0];
				B = markerLocations[1];
				C = markerLocations[2];

			}
			else if (BtoC < AtoB && BtoC < AtoC) {
				A = markerLocations[1];
				B = markerLocations[2];
				C = markerLocations[0];
			}
			else if (AtoC < AtoB && AtoC < BtoC) {
				A = markerLocations[0];
				B = markerLocations[2];
				C = markerLocations[1];
			}
			/*
			drawMarker(newundistort, A, Scalar(180, 180, 0), 0, 20, 4);
			drawMarker(newundistort, B, Scalar(180, 0, 180), 0, 20, 4);
			drawMarker(newundistort, C, Scalar(0, 180, 180), 0, 20, 4);
			*/
			Point halfway = Point((A.x + B.x) / 2, (A.y + B.y) / 2); //finds the points halfway between A and B
			line(newundistort, halfway, C, Scalar(0, 128, 255), 3);
			double y = double(C.y - halfway.y);
			double x = double(C.x - halfway.x);
			double angle = atan2(y, x) * (180 / PI); //This finds the angle the triangle is facing
			if (angle < 0) {
				angle = angle + 360;
			}
			double centerx = (A.x + B.x + C.x) / 3;
			double centery = (A.y + B.y + C.y) / 3;
			Point center = Point(centerx, centery); //This finds the centroid of the triangle and save it as a point
			drawMarker(newundistort, center, Scalar(0, 128, 255), 3);
			String str_centerx = to_string(center.x); //Puts all the data intro a string
			String str_centery = to_string(center.y);
			String str_angle = to_string(angle);
			String datafile = str_centerx + " " + str_centery + "  " + str_angle;
			cv::putText(newundistort, datafile, Point(10, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 180, 180),4);
			try {

				socket.send_to(buffer(datafile), remote);
			}
			catch (const boost::system::system_error& ex) {
			}
		}
		cv::createTrackbar("Threshhold Value", "frame", &threshhold, 255);
		cv::createTrackbar("THRESHHOLD IMAGE", "frame", &onInt, 1);
		
		
		
		
		
		try {
			if (onInt == 0) {
				imshow("frame", newundistort);
				if (saveimage) {
					imwrite("test.jpg", newundistort);
					saveimage = false;
				}
				
			}
			else {
				imshow("frame", gpuimage);
			}

		}
		catch (exception e) {
			//cout << e.what();
		}
		long double e2 = getTickCount();
		long double time = (e2 - e1) / getTickFrequency();

		cout << 1 / time << endl;
		key = cv::waitKey(1);
		
		
	}

	camera.Disconnect();


}




void arucodetect() {
	Camera camera;
	CameraInfo camInfo;


	camera.Connect(0);
	camera.GetCameraInfo(&camInfo);
	camera.StartCapture();
	char key = 0;

	Image rawImage;
	Image rgbImage;
	Mat original;

	namedWindow("frame", WINDOW_KEEPRATIO);
	resizeWindow("frame", 600, 800);
	UMat gpuimage;

	int thresh1 = 68;
	int thresh2 = 1256;
	int thresh3 = 68;
	int thresh4 = 1256;
	int dilateint = 2;
	int erodeeint = 2;


	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);

	while (key != 'q') {

		double e1 = getTickCount();
		camera.RetrieveBuffer(&rawImage);
		rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);
		unsigned int rowBytes = (double)rgbImage.GetReceivedDataSize() / (double)rgbImage.GetRows();
		original = Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);



		cvtColor(original, gpuimage, COLOR_RGB2HSV);

		//resize(original, gpuimage, Size(), 0.5, 0.5, INTER_LANCZOS4);

		Canny(original, gpuimage, thresh1, thresh2);
		dilate(gpuimage, gpuimage, getStructuringElement(MORPH_RECT, Size(dilateint, dilateint)));
		threshold(gpuimage, gpuimage, thresh3, thresh4, THRESH_BINARY_INV);
		erode(gpuimage, gpuimage, getStructuringElement(MORPH_RECT, Size(erodeeint, erodeeint)));


		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f> > corners;






		// if at least one marker detected

		if (ids.size() > 0)

			/*
			for (int i = 0; i < corners[0].size(); i++) {
				corners[0][i] = corners[0][i] * 2;
			}
			*/
			aruco::drawDetectedMarkers(gpuimage, corners, ids);






		cv::createTrackbar("thresh1", "frame", &thresh1, 3000);
		cv::createTrackbar("thresh2", "frame", &thresh2, 3000);
		cv::createTrackbar("dilate", "frame", &dilateint, 10);
		cv::createTrackbar("erode", "frame", &erodeeint, 10);
		cv::createTrackbar("thresh3", "frame", &thresh3, 3000);
		cv::createTrackbar("thresh4", "frame", &thresh4, 3000);
		//cout << time << endl;

		try {

			imshow("frame", gpuimage);

		}
		catch (exception e) {
			cout << e.what();
		}

		key = cv::waitKey(1);

	}
	camera.Disconnect();

}
void calibrate() {

	vector<Mat> chessboardImages; //Holds all of our input images for running calibration
	vector<string> nameofImage; //holds the name of each image in correspondance to the chessboardImages vector

	const int numofPics = 77; //specify how many images are in the folder; Should follow the format for naming: "<number>.jpg" starting at <number> = 1
	const Size boardSize = Size(9, 6);  //The size of the board used
	vector<vector<Point2f>> boardImagePoints; //holds the points containg the location of the chessboard intersections
	vector<vector<Point2f>> boardObjectPoints;

	//This loop adds all images in the folder to the MAT vector
	for (int i = 1; i < numofPics + 1; i++) {
		string filename;
		filename.append(to_string(i));
		filename.append(".jpg");
		nameofImage.push_back(filename);  //pushes filename to the vector of strings
		chessboardImages.push_back(imread(filename));  //pushes MAT object to the vector

		//imshow(filename, chessboardImages[i - 1]);
		//waitKey(0);
	}


	//This loop pushes the found chessboard points to the boardpoints vector of vectors and also draws the points and display them for each picture
	while (!chessboardImages.empty()) {
		vector<Point2f> pointBuf;
		Mat temp = chessboardImages.back(); //our temporary matrix that holds each image when popped from the stack
		String tempFilename = nameofImage.back();

		//cvtColor(temp, temp, COLOR_BGR2GRAY);  //converts our image to grayscale


		bool found = findChessboardCorners(temp, boardSize, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);  //if corners are found, it passes the next if statement

		if (found) {

			//cornerSubPix(temp, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(0, 30, 0.1));  //cornersubpix allows for a higher precision in finding the corners
			boardImagePoints.push_back(pointBuf);
			drawChessboardCorners(temp, boardSize, boardImagePoints.back(), found);  //draws the corners on the given MAT object
			namedWindow(tempFilename, WINDOW_KEEPRATIO);  //makes windows resizable
			resizeWindow(tempFilename, 900, 500);  //sets size of window so its not huge
			imshow(tempFilename, temp); //display windows with drawn chessboard 
			waitKey(0); //waits until key is pressed and closes window
			destroyAllWindows();
			boardImagePoints.pop_back(); //pops out the boardpoints from the stack to contineu to the next image
		}


		chessboardImages.pop_back();  //pops out the last MAT object so the loop can continue to the next image
		nameofImage.pop_back(); //pops out last string to keep up with the chessboardImage stack
	}







}
int main()
{
	showFrames();
	return 0;


}