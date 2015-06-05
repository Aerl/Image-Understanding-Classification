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
	DecisionMaker GetClassification;

	std::vector<std::string> folders;
	folders.push_back("accordion");
	folders.push_back("airplanes");
	folders.push_back("anchor");
	folders.push_back("ant");
	folders.push_back("barrel");
	folders.push_back("bass");
	folders.push_back("beaver");
	folders.push_back("binocular");
	folders.push_back("bonsai");



	LoadImages.LoadImagesFromSubfolders(folders);
	//LoadImages.LoadImages();
	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	LoadImages.getTrainingData(trainingImages, trainingLabels);
	LoadImages.getTestData(testImages, testLabels);

	std::vector<std::vector< cv::Mat >> FeatureVectors = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeHOGFeatures(trainingImages, FeatureVectors);

	std::vector<std::vector< cv::Mat >> SURFTrain = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeSURFFeatures(trainingImages, SURFTrain);

	std::vector<std::vector< cv::Mat >> SURFTest = std::vector<std::vector< cv::Mat>>(testImages.size());
	GetFeatures.computeSURFFeatures(testImages, SURFTest);

	std::vector<int> classificationResults = std::vector<int>(testImages.size());
	GetFeatures.MakeDecisionFLANN(SURFTrain, SURFTest, trainingLabels, classificationResults);

	std::vector<std::string> classNames;
	LoadImages.getClassNames(classNames);
	int NumberOfSamples;
	LoadImages.getSampleSize(NumberOfSamples);
	int NumberOfClasses = classNames.size();

	EvaluationUnit GetEvaluation(testLabels,NumberOfClasses,NumberOfSamples);

	double percent = GetEvaluation.EvaluateResultSimple(classificationResults);
	std::cout << "Simple Percentage: " + std::to_string(percent) << std::endl;

	std::vector<double> classPercentage;
	std::vector<std::vector<int>> statistics;
	GetEvaluation.EvaluateResultComplex(classificationResults, classPercentage, statistics);
	for (int i = 0; i < classPercentage.size(); i++)
	{
		std::cout << "Complex Percentage Class " + std::to_string(i) + " : " + std::to_string(classPercentage[i]) << std::endl;
	}

	return 0;
}

