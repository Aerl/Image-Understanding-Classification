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
	//folders.push_back("../101_ObjectCategories/octopus");
	folders.push_back("../101_ObjectCategories/scissors");
	//folders.push_back("../101_ObjectCategories/sunflower");
	//folders.push_back("../101_ObjectCategories/wrench");
	//folders.push_back("../101_ObjectCategories/helicopter");
	//folders.push_back("../101_ObjectCategories/platypus");
	//folders.push_back("../101_ObjectCategories/airplanes");


	// minimum class-size = 31
	LoadImages.LoadImagesFromSubfolders(folders, 31);

	const std::vector<cv::Mat> images = LoadImages.getImages();

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);
	SurfDescriptorExtractor extractor;

	std::vector<std::vector<KeyPoint>> keypoints_training;
	std::vector<Mat> descriptors;
	Mat descriptor;
	int index = 0;

	for (Mat image : images)
	{
		keypoints_training.push_back(std::vector<KeyPoint>());
		detector.detect(image, keypoints_training.at(index));
		extractor.compute(image, keypoints_training.at(index), descriptor);
		descriptors.push_back(descriptor);
		index++;
	}



	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors.at(0), descriptors.at(1), matches);

	//-- Draw matches
	Mat img_matches;
	drawMatches(images.at(0), keypoints_training.at(0), images.at(1), keypoints_training.at(1), matches, img_matches);

	//-- Show detected matches
	imshow("Matches", img_matches);

	waitKey(1000000);


	//-- Draw keypoints
	Mat img_keypoints;

	for (int iter = 0; iter < keypoints_training.size(); ++iter)
	{
		drawKeypoints(images.at(iter), keypoints_training.at(iter), img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//-- Show detected (drawn) keypoints
		imshow("Keypoints 1", img_keypoints);
		waitKey(1000);
	}

	return 0;
}