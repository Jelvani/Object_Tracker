///////////////////////////////////////////////////////////////////
// Copyright (C) 2019, Alborz Jelvani, all rights reserved
// Libraries used that are NOT mine include Flycapture for
// receiving gige camera stream, Boost for network communication,
// and OpenCV for image processing.
///////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <FlyCapture2.h>
#include <string>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "boost/asio.hpp"

#define PI 3.14159265358979323846
#define PORT "10090"		 //UDP port to send data to
#define IP "192.168.200.192" //IP to send data to
#define RESOLUTION 1024, 768 //Resolution from camera

using namespace cv;
using namespace FlyCapture2;
using namespace std;
using namespace boost::asio;

void showFrames()
{
	Camera camera;
	CameraInfo camInfo;
	camera.Connect(0);
	camera.GetCameraInfo(&camInfo);
	camera.StartCapture();
	Image rawImage;
	Image rgbImage;
	UMat gpuimage;
	char key = 0;
	int threshhold = 250;
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

	float intrinsic[3][3] = {5.266478108199441e+02, 0, 5.090669900210755e+02, 0, 5.266442917480177e+02, 4.056045815230510e+02, 0, 0, 1}; //Camera intrinsic parameters
	float distortion[1][4] = {-0.3452, 0.1265, 0, 0};																					 //Camera distortion parameters
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, intrinsic);
	Mat distortion_coeffs = Mat(1, 4, CV_32FC1, distortion);
	UMat newundistort;
	Mat original;
	Size imageSize = Size(RESOLUTION);
	Mat R;
	Mat mapx = Mat(imageSize, CV_32FC1);
	Mat mapy = Mat(imageSize, CV_32FC1);
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, imageSize, CV_32FC1, mapx, mapy);
	auto saveimage = true;

	while (key != 'q')
	{ //Q to quit
		long double e1 = getTickCount();
		camera.RetrieveBuffer(&rawImage);
		rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);

		original = Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData());

		remap(original, newundistort, mapx, mapy, INTER_LINEAR);
		cvtColor(newundistort, gpuimage, COLOR_BGR2GRAY);

		threshold(gpuimage, gpuimage, threshhold, 255, THRESH_BINARY);

		findContours(gpuimage, contourpoints, RETR_TREE, CHAIN_APPROX_SIMPLE);
		if (contourpoints.size() > 2)
		{ //If all 3 markers are found
			vector<vector<Point>> savePoint1;

			for (int j = 0; j < 3; j++)
			{ //puts only the 3 biggest contours into final vector, this acts as a filter
				int index = 0;
				double area = contourArea(contourpoints[0], false);
				for (int i = 0; i < contourpoints.size(); i++)
				{
					if (contourArea(contourpoints[i]) > area)
					{
						area = contourArea(contourpoints[i]);
						index = i;
					}
				}
				savePoint1.push_back(contourpoints[index]);
				contourpoints.erase(contourpoints.begin() + index);
			}

			vector<Point> markerLocations;
			for (int i = 0; i < 3; i++)
			{ //this gets and saves the point location for each contour by doing a bounded rectangle average
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

			if (AtoB < BtoC && AtoB < AtoC)
			{ //this finds and assigns point A and B as the 2 closest points and C as the farthest
				A = markerLocations[0];
				B = markerLocations[1];
				C = markerLocations[2];
			}
			else if (BtoC < AtoB && BtoC < AtoC)
			{
				A = markerLocations[1];
				B = markerLocations[2];
				C = markerLocations[0];
			}
			else if (AtoC < AtoB && AtoC < BtoC)
			{
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
			if (angle < 0)
			{
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
			cv::putText(newundistort, datafile, Point(10, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 180, 180), 4);
			try
			{

				socket.send_to(buffer(datafile), remote);
			}
			catch (const boost::system::system_error &ex)
			{
			}
		}
		cv::createTrackbar("Threshhold Value", "frame", &threshhold, 255);
		cv::createTrackbar("THRESHHOLD IMAGE", "frame", &onInt, 1);

		try
		{
			if (onInt == 0)
			{
				imshow("frame", newundistort);
				if (saveimage)
				{
					imwrite("track.jpg", newundistort);
					saveimage = false;
				}
			}
			else
			{
				imshow("frame", gpuimage);
			}
		}
		catch (exception e)
		{
			cout << e.what();
		}
		key = cv::waitKey(1);
	}
	camera.Disconnect();
}
}
int main()
{
	showFrames();
	return 0;
}