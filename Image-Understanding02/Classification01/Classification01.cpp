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
#include <string>
#include <vector>


int _tmain(int argc, _TCHAR* argv[])
{
	//std::string path("D:/GitHub/Image-Understanding-Classification/Image-Understanding02/101_ObjectCategories");
	std::string path("../101_ObjectCategories");
	ImageLoader LoadImages = ImageLoader(path);
	FeatureExtractor GetFeatures;

	std::vector<std::string> folders;
	folders.push_back("accordion");
	folders.push_back("sunflower");
	folders.push_back("wrench");
	folders.push_back("helicopter");
	folders.push_back("platypus");


	LoadImages.LoadImagesFromSubfolders(folders);
	//LoadImages.LoadImages();
	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;
	LoadImages.getTrainingData(trainingImages, trainingLabels);

	std::vector<std::vector< cv::Mat >> FeatureVectors = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeHOGFeatures(trainingImages, FeatureVectors);

	//cv::Mat Feature = FeatureVectors[3][0];

	//std::cout << "  Feature: " + std::to_string(FeatureVectors.size()) << std::endl;


	//namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

	//imshow("Show Images", FeatureVectors[3][0]);

	for (std::vector<cv::Mat>::iterator iter = trainingImages.begin(); iter != trainingImages.end(); ++iter)
	{

		namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

		imshow("Show Images", *iter);
		//std::cout << "  class: " + iter->category << std::endl;
		cv::waitKey(300);
	}
	/*
	std::vector<Image> testImages = LoadImages.getTestImages();

	for (std::vector<Image>::iterator iter = testImages.begin(); iter != testImages.end(); ++iter)
	{

		namedWindow("Show Images", cv::WINDOW_AUTOSIZE);

		imshow("Show Images", iter->data);
		std::cout << "  class: " + iter->category << std::endl;
		cv::waitKey(300);
	}*/

	return 0;
}

