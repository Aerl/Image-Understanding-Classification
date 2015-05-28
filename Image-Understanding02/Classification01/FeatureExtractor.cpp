#include "stdafx.h"
#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{

}

FeatureExtractor::Parameters::Parameters()
{
	this->WindowX = 144;
	this->WindowY = 144;
}

FeatureExtractor::~FeatureExtractor()
{

}

void FeatureExtractor::computeHOGFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectors)
{
	cv::HOGDescriptor hog;
	hog.winSize = cv::Size(this->parameters.WindowX, this->parameters.WindowY);
	cv::Mat gray;
	std::vector< cv::Point > location;
	std::vector< float > descriptors;
	int Size = int(Images.size());

	for (int iter = 0; iter < Size; ++iter)
	{
		cv::cvtColor(Images[iter], gray, cv::COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);
		std::cout << "  Feature Size: " + std::to_string(descriptors.size()) << std::endl;
		FeatureVectors[iter].push_back(cv::Mat(descriptors).clone());
	}
}


