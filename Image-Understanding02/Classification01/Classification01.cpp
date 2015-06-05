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
	folders.push_back("sunflower");
	folders.push_back("wrench");
	folders.push_back("helicopter");
	folders.push_back("platypus");


	LoadImages.LoadImagesFromSubfolders(folders);
	//LoadImages.LoadImages();
	std::vector<cv::Mat> trainingImages;
	std::vector<int> trainingLabels;

	std::vector<cv::Mat> testImages;
	std::vector<int> testLabels;

	std::vector<int> classificationResults;

	LoadImages.getTrainingData(trainingImages, trainingLabels);
	LoadImages.getTrainingData(testImages, testLabels);

	std::vector<std::vector< cv::Mat >> FeatureVectors = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeHOGFeatures(trainingImages, FeatureVectors);

	std::vector<std::vector< cv::Mat >> SURFTrain = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeSURFFeatures(trainingImages, SURFTrain);
	std::vector<std::vector< cv::Mat >> SURFTest = std::vector<std::vector< cv::Mat>>(trainingImages.size());
	GetFeatures.computeSURFFeatures(trainingImages, SURFTest);

	GetClassification.MakeDecisionFLANN(SURFTrain, SURFTest, trainingLabels, classificationResults);




	//int num_files = trainingImages.size();
	//int img_area = 150*150;
	//cv::Mat labels(num_files, 1, CV_32FC1);
	//cv::Mat training_mat(num_files, img_area, CV_32FC1);
	//int labelIndex = 0;

	//// reshape Images to one Mat for the SVM
	//for (cv::Mat img_mat : trainingImages)
	//{
	//	int ii = 0;
	//	for (int i = 0; i < img_mat.rows; i++) 
	//	{
	//		for (int j = 0; j < img_mat.cols; j++) 
	//		{
	//			training_mat.at<float>(labelIndex,ii++) = img_mat.at<uchar>(i, j);
	//		}
	//	}
	//	labels.at<float>(labelIndex) = trainingLabels.at(labelIndex);
	//	labelIndex++;
	//}
	//
	//// set up SVM parameters
	//CvSVMParams params;
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::LINEAR;
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	////...etc

	//// set up Support Vector Machine for training and classification 
	//cv::SVM svm; 
	//svm.train(training_mat, labels, cv::Mat(), cv::Mat(), params);

	//// svm.predict to classify an image
	//std::vector<cv::Mat> testData;
	//std::vector<int> testLabels;
	//LoadImages.getTrainingData(testData, testLabels);

	//cv::Mat test2 = training_mat.row(18);

	//int p = svm.predict(test2);





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

