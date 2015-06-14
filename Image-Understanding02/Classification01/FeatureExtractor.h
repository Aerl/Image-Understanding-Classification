#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "Image.h"
#include <vector>

#pragma once
class FeatureExtractor
{
	struct Parameters
	{
		int WindowX;
		int WindowY;
		int histSize;
		//! Default constructor ensuring that all variables are initialized.
		Parameters();
	};
	Parameters parameters;
public:
	FeatureExtractor();
	//FeatureExtractor(std::vector<Image> Images);
	~FeatureExtractor();
	void computeHOGFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectors);
	void computeSURFFeatures(std::vector<cv::Mat> &TrainImages, std::vector<cv::Mat> &FeatureVectorsSURFUnclustered, int &DictionarySize);
	void computeColorFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectorsColor);
	void MakeDecisionFLANN(std::vector<std::vector< cv::Mat >> &featuresTrain, std::vector<std::vector< cv::Mat >> &featuresTest, std::vector<int> &trainingLabels, std::vector<int> &classificationResults);
	void getBagOfWords(std::vector<cv::Mat> &TestImages, std::vector<cv::Mat> &FeatureVectorsSURFUnclustered, std::vector<std::vector< cv::Mat >> &clusteredFeatures, int &DictionarySize);
};

