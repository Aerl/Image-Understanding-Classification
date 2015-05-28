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

	std::vector<std::vector<KeyPoint>> keypoints;
	std::vector<Mat> descriptors;
	Mat descriptor;
	int index = 0;

	for (Mat image : images)
	{
		keypoints.push_back(std::vector<KeyPoint>());
		detector.detect(image, keypoints.at(index));
		extractor.compute(image, keypoints.at(index), descriptor);
		descriptors.push_back(descriptor);
		index++;
	}

	waitKey(0);

	return 0;
}