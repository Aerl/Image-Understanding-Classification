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
	//folders.push_back("airplanes");
	folders.push_back("anchor");
	//folders.push_back("ant");
	folders.push_back("barrel");
	folders.push_back("bass");
	//folders.push_back("beaver");
	folders.push_back("binocular");
	folders.push_back("bonsai");



	//LoadImages.LoadImagesFromSubfolders(folders);
	LoadImages.LoadImages();
	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	LoadImages.getTrainingData(trainingImages, trainingLabels);
	LoadImages.getTestData(testImages, testLabels);

	std::vector<std::vector< cv::Mat >> FeatureVectors = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeHOGFeatures(trainingImages, FeatureVectors);
	GetFeatures.computeColorFeatures(trainingImages, FeatureVectors);

	//GetClassification.TrainSVM(FeatureVectors, trainingLabels);

	//std::vector<std::string> classNames;
	//LoadImages.getClassNames(classNames);
	//int NumberOfClasses = classNames.size();


	//std::vector<int> classificationResults = std::vector<int>(testImages.size());
	

	GetClassification.TrainSVM(FeatureVectors, trainingLabels);
	// svm.predict to classify an image
	

	FeatureVectors = std::vector<std::vector< cv::Mat>>(testImages.size());

	GetFeatures.computeHOGFeatures(testImages, FeatureVectors);
	GetFeatures.computeColorFeatures(testImages, FeatureVectors);

	std::cout << "FeatureVectors: " + std::to_string(FeatureVectors.size()) << std::endl;

	std::vector<int> Results;
	GetClassification.PredictSVM(FeatureVectors, Results);

	std::cout << "Results: " + std::to_string(Results.size()) << std::endl;


	std::vector<std::string> classNames;
	LoadImages.getClassNames(classNames);
	int NumberOfSamples;
	LoadImages.getSampleSize(NumberOfSamples);
	int NumberOfClasses = classNames.size();

	EvaluationUnit GetEvaluation(testLabels, NumberOfClasses, NumberOfSamples);

	double percent = GetEvaluation.EvaluateResultSimple(Results);
	std::cout << "Simple Percentage: " + std::to_string(percent) << std::endl;

	std::vector<double> classPercentage;
	std::vector<std::vector<int>> statistics;
	GetEvaluation.EvaluateResultComplex(Results, classPercentage, statistics);
	for (int i = 0; i < classPercentage.size(); i++)
	{
		std::cout << "Complex Percentage Class " + std::to_string(i) + " : " + std::to_string(classPercentage[i]) << std::endl;
	}



	return 0;



}

