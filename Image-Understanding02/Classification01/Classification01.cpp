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

	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	std::vector<std::vector< cv::Mat >> FeatureVectorsTraining;
	std::vector<std::vector< cv::Mat >> FeatureVectorsTest;

	std::vector<int> ResultsTest;
	std::vector<int> ResultsTraining;
	
	std::vector<std::string> classNames;
	int NumberOfSamples;
	int NumberOfClasses;

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
	

	for (int i = 0; i < 5; ++i)
	{
		//LoadImages.LoadImagesFromSubfolders(folders);
		LoadImages.LoadImages();
		LoadImages.getTrainingData(trainingImages, trainingLabels);
		LoadImages.getTestData(testImages, testLabels);

		FeatureVectorsTraining.clear();
		FeatureVectorsTraining.resize(trainingImages.size());

		GetFeatures.computeHOGFeatures(trainingImages, FeatureVectorsTraining);
		GetFeatures.computeColorFeatures(trainingImages, FeatureVectorsTraining);

		GetClassification.TrainSVM(FeatureVectorsTraining, trainingLabels);

		GetClassification.PredictSVM(FeatureVectorsTraining, ResultsTraining);

		FeatureVectorsTest.clear();
		FeatureVectorsTest.resize(testImages.size());

		GetFeatures.computeHOGFeatures(testImages, FeatureVectorsTest);
		GetFeatures.computeColorFeatures(testImages, FeatureVectorsTest);

		GetClassification.PredictSVM(FeatureVectorsTest, ResultsTest);
		
		LoadImages.getClassNames(classNames);
		NumberOfClasses = classNames.size();
		LoadImages.getSampleSize(NumberOfSamples);

		EvaluationUnit GetTrainingEvaluation(trainingLabels, NumberOfClasses, NumberOfSamples);
		double TrainingPercent = GetTrainingEvaluation.EvaluateResultSimple(ResultsTraining);
		std::cout << " Training Data Simple Percentage: " + std::to_string(TrainingPercent) << std::endl;

		EvaluationUnit GetTestEvaluation(testLabels, NumberOfClasses, NumberOfSamples);
		double TestPercent = GetTestEvaluation.EvaluateResultSimple(ResultsTest);
		std::cout << " Test Data Simple Percentage: " + std::to_string(TestPercent) << std::endl;

		

	}

	
	
	

	/*std::vector<double> classPercentage;
	std::vector<std::vector<int>> statistics;
	GetEvaluation.EvaluateResultComplex(Results, classPercentage, statistics);
	for (int i = 0; i < classPercentage.size(); i++)
	{
		std::cout << "Complex Percentage Class " + std::to_string(i) + " : " + std::to_string(classPercentage[i]) << std::endl;
	}
*/


	return 0;



}

