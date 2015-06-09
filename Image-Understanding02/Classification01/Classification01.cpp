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

	LoadImages.LoadImagesFromSubfolders(folders);
	//LoadImages.LoadImages();
	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	LoadImages.getTrainingData(trainingImages, trainingLabels);
	LoadImages.getTestData(testImages, testLabels);

	std::vector<std::vector< cv::Mat >> unclusteredSURFFeatures = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	std::vector<std::vector< cv::Mat >> FeatureVectors = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	//GetFeatures.computeHOGFeatures(trainingImages, FeatureVectors);
	//GetFeatures.computeColorFeatures(trainingImages, FeatureVectors);
	GetFeatures.computeSURFFeatures(trainingImages, unclusteredSURFFeatures);

	int row1 = 1000;
	for (std::vector<cv::Mat> fvector : unclusteredSURFFeatures)
	{
		for (cv::Mat feature : fvector)
		{
			if (row1 > feature.rows)
			{
				row1 = feature.rows;
			}
		}
	}


	std::vector<std::vector< cv::Mat>> FeatureVectorsTest = std::vector<std::vector< cv::Mat>>(testImages.size());
	std::vector<std::vector< cv::Mat>> unclusteredSURFFeaturesTest = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeSURFFeatures(testImages, unclusteredSURFFeaturesTest);

	int row2 = 1000;
	for (std::vector<cv::Mat> fvector : unclusteredSURFFeaturesTest)
	{
		for (cv::Mat feature : fvector)
		{
			if (row2 > feature.rows)
			{
				row2 = feature.rows;
			}
		}
	}

	int dictionarySize = 0;
	if (row1 > row2)
	{
		dictionarySize = row2;
	}
	else
	{
		dictionarySize = row1;
	}

	GetFeatures.getBagOfWords(testImages, unclusteredSURFFeaturesTest, FeatureVectorsTest, dictionarySize);
	GetFeatures.getBagOfWords(trainingImages, unclusteredSURFFeatures, FeatureVectors, dictionarySize);

	GetClassification.TrainSVM(FeatureVectors, trainingLabels);

	CvRTrees trees;

	//float priors[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	//CvRTParams params = CvRTParams(25, // max depth
	//	5, // min sample count
	//	0, // regression accuracy: N/A here
	//	false, // compute surrogate split, no missing data                                    
	//	15, // max number of categories (use sub-optimal algorithm for larger numbers)
	//	priors, // the array of priors
	//	false,  // calculate variable importance
	//	4,       // number of variables randomly selected at node and used to find the best split(s).
	//	100,   // max number of trees in the forest
	//	0.01f,             // forrest accuracy
	//	CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
	//	);

	cv::Mat reshapedFeatureVectors;
	GetClassification.ReshapeFeatures(FeatureVectors, reshapedFeatureVectors);
	cv::Mat reshapedTrainingLabels;
	GetClassification.ReshapeLabels(trainingLabels, reshapedTrainingLabels);

	cv::Mat reshapedFeatureVectorsTest;
	GetClassification.ReshapeFeatures(FeatureVectorsTest, reshapedFeatureVectorsTest);
	cv::Mat reshapedTestLabels;
	GetClassification.ReshapeLabels(testLabels, reshapedTestLabels);


	trees.train(reshapedFeatureVectors, CV_ROW_SAMPLE, reshapedTrainingLabels);

	trees.predict(reshapedFeatureVectorsTest);
		


	// svm.predict to classify an image
	
	//GetFeatures.computeHOGFeatures(testImages, FeatureVectors);
	//GetFeatures.computeColorFeatures(testImages, FeatureVectors);

	//std::cout << "FeatureVectors: " + std::to_string(FeatureVectors.size()) << std::endl;

	std::vector<int> Results;
	GetClassification.PredictSVM(FeatureVectorsTest, Results);

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

