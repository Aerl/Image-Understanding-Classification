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

public:
	DecisionMaker();
	DecisionMaker(int xDim, int yDim);
	~DecisionMaker();
	void MakeDecisionSVM(std::vector<std::vector< cv::Mat >> &features, std::vector<int> &trainingLabels, std::vector<int> &classificationResults);
};

