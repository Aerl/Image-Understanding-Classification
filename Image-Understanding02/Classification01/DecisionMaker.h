#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>


class DecisionMaker
{

	struct Parameters
	{
		int ROIx;
		int ROIy;
		//! Default constructor ensuring that all variables are initialized.
		Parameters();
	};

	Parameters parameters;
	cv::SVM svm;
	CvRTrees rtree;
	cv::PCA pca;
public:
	//Constructor
	DecisionMaker();
	DecisionMaker(int xDim, int yDim);
	//Dstructor
	~DecisionMaker();
	//SVMs
	void TrainSVM(cv::Mat &FeatureVectors, std::vector<int> &trainingLabels);
	void PredictSVM(cv::Mat &FeatureVectors, std::vector<int> &ClassificationResults);
	//Random Trees
	void TrainRandomTrees(cv::Mat &FeatureVectors, std::vector<int> &trainingLabels);
	void PredictRandomTrees(cv::Mat &FeatureVectors, std::vector<int> &ClassificationResults);;
	//Feature Reduction
	void reduceFeaturesPCA(cv::Mat &Features, cv::Mat &ReducedFeatures);
	//Reshaping
	void ReshapeLabels(cv::Mat &Labels, std::vector<int> &ReshapedLabels);
	void ReshapeLabels(std::vector<int> &Labels, cv::Mat &ReshapedLabels);
	void ReshapeFeatures(std::vector<std::vector< cv::Mat >> &FeatureVectors, cv::Mat &ReshapedFeatures);

	void DecisionMaker::constructPCA(cv::Mat &Features);
	};


