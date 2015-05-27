#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "Image.h"
#include <vector>

#pragma once
class FeatureExtractor
{
	struct Parameters
	{
		int WindowX;
		int WindowY;
		//! Default constructor ensuring that all variables are initialized.
		Parameters();
	};
	Parameters parameters;
public:
	FeatureExtractor();
	//FeatureExtractor(std::vector<Image> Images);
	~FeatureExtractor();
	void computeHOGFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectors);
};

