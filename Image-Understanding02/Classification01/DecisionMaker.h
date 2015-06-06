#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

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

public:
	DecisionMaker();
	DecisionMaker(int xDim, int yDim);
	~DecisionMaker();
	void TrainSVM(std::vector<std::vector< cv::Mat >> &FeatureVectors, std::vector<int> &trainingLabels);
	void PredictSVM(std::vector<std::vector< cv::Mat >> &FeatureVectors, std::vector<int> &ClassificationResults);
	void MakeDecisionFLANN(std::vector<std::vector< cv::Mat >> &featuresTrain, std::vector<std::vector< cv::Mat >> &featuresTest, std::vector<int> &trainingLabels, std::vector<int> &classificationResults);
private:
	void ReshapeLabels(std::vector<int> &Labels, cv::Mat &ReshapedLabels);
	void ReshapeLabels(cv::Mat &Labels, std::vector<int> &ReshapedLabels);
	void ReshapeFeatures(std::vector<std::vector< cv::Mat >> &FeatureVectors, cv::Mat &ReshapedFeatures);
};

