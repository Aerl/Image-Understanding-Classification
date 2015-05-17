// Classification03.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "ImageLoader.h"
#include<string>

//using namespace cv;
//using namespace std;


int _tmain(int argc, _TCHAR* argv[])
{
	std::string path("../101_ObjectCategories");
	ImageLoader LoadImages = ImageLoader(path);
	//LoadImages.LoadAllImagesFromSubfolder("accordion");
	//LoadImages.LoadAllImages();

	std::vector<std::string> folders;
	folders.push_back("../101_ObjectCategories/accordion");
	folders.push_back("../101_ObjectCategories/crab");
	folders.push_back("../101_ObjectCategories/garfield");
	folders.push_back("../101_ObjectCategories/octopus");
	folders.push_back("../101_ObjectCategories/scissors");
	folders.push_back("../101_ObjectCategories/sunflower");
	folders.push_back("../101_ObjectCategories/wrench");
	LoadImages.LoadAllImagesFromSubfolders(folders);

	std::vector<cv::Mat> images = LoadImages.getImages();

	for (std::vector<cv::Mat>::iterator iter = images.begin(); iter != images.end(); ++iter)
	{

		namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

		imshow("Show Images", *iter);
		cv::waitKey(100);
	}

	return 0;
}

