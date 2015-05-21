#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <string>
#include "stdafx.h"
#include "ImageLoader.h"


using namespace cv;

void readme();

/** @function main */
int main(int argc, char** argv)
{

	std::string path("../101_ObjectCategories");

	ImageLoader LoadImages = ImageLoader(path);

	std::vector<std::string> folders;
	//folders.push_back("../101_ObjectCategories/accordion");
	//folders.push_back("../101_ObjectCategories/crab");
	//folders.push_back("../101_ObjectCategories/garfield");
	folders.push_back("../101_ObjectCategories/octopus");
	folders.push_back("../101_ObjectCategories/scissors");
	//folders.push_back("../101_ObjectCategories/sunflower");
	//folders.push_back("../101_ObjectCategories/wrench");
	folders.push_back("../101_ObjectCategories/helicopter");
	//folders.push_back("../101_ObjectCategories/platypus");
	folders.push_back("../101_ObjectCategories/airplanes");

	LoadImages.LoadImagesFromSubfolders(folders, 5);

	std::vector<cv::Mat> images = LoadImages.getImages();

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<std::vector<KeyPoint>> keypoints;
	int index = 0;

	for (Mat image : images)
	{
		keypoints.push_back(std::vector<KeyPoint>());
		detector.detect(image, keypoints.at(index));
		index++;
	}



	//-- Draw keypoints
	Mat img_keypoints;

	for (int iter = 0; iter < keypoints.size(); ++iter)
	{
		drawKeypoints(images.at(iter), keypoints.at(iter), img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//-- Show detected (drawn) keypoints
		imshow("Keypoints 1", img_keypoints);
		waitKey(1000);
	}

	return 0;
}