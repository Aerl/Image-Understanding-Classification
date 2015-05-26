// Classification01.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ImageLoader.h"
#include "DecisionMaker.h"
#include "EvaluationUnit.h"
#include "FeatureExtractor.h"
#include "Image.h"
#include<string>


int _tmain(int argc, _TCHAR* argv[])
{
	std::string path("../101_ObjectCategories");
	ImageLoader LoadImages = ImageLoader(path);

	std::vector<std::string> folders;
	folders.push_back("accordion");
	//folders.push_back("../101_ObjectCategories/crab");
	//folders.push_back("../101_ObjectCategories/garfield");
	//folders.push_back("octopus");
	//folders.push_back("scissors");
	folders.push_back("sunflower");
	//folders.push_back("../101_ObjectCategories/wrench");
	folders.push_back("helicopter");
	//folders.push_back("../101_ObjectCategories/platypus");
	//folders.push_back("../101_ObjectCategories/airplanes");


	LoadImages.LoadImagesFromSubfolders(folders);
	//LoadImages.LoadImages();
	std::vector<Image> trainingImages = LoadImages.getTrainingImages();

	for (std::vector<Image>::iterator iter = trainingImages.begin(); iter != trainingImages.end(); ++iter)
	{

		namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

		imshow("Show Images", iter->data);
		std::cout << "  class: " + iter->category << std::endl;
		cv::waitKey(300);
	}

	std::vector<Image> testImages = LoadImages.getTestImages();

	for (std::vector<Image>::iterator iter = testImages.begin(); iter != testImages.end(); ++iter)
	{

		namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

		imshow("Show Images", iter->data);
		std::cout << "  class: " + iter->category << std::endl;
		cv::waitKey(300);
	}

	return 0;
}

