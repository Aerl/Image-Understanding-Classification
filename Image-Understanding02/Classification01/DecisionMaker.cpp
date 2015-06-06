#include "stdafx.h"
#include "DecisionMaker.h"
#include "FeatureExtractor.h"

DecisionMaker::Parameters::Parameters()
{
	this->ROIx = 150;
	this->ROIy = 150;
}

DecisionMaker::DecisionMaker()
{
}

DecisionMaker::DecisionMaker(int xDim, int yDim)
{
	this->parameters.ROIx = xDim;
	this->parameters.ROIy = yDim;
}


DecisionMaker::~DecisionMaker()
{
}

void DecisionMaker::TrainSVM(std::vector<std::vector< cv::Mat >> &FeatureVectors, std::vector<int> &Labels)
{
	cv::Mat ReshapedFeatures;
	cv::Mat ReshapedLabels;
	ReshapeLabels(Labels, ReshapedLabels);
	ReshapeFeatures(FeatureVectors, ReshapedFeatures);

	// set up SVM parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.C = 0.5;
	params.gamma = 0.5;
	params.nu = 0.5;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

	// set up Support Vector Machine for training and classification 
	this->svm.train(ReshapedFeatures, ReshapedLabels, cv::Mat(), cv::Mat(), params);
}

void DecisionMaker::PredictSVM(std::vector<std::vector< cv::Mat >> &FeatureVectors, std::vector<int> &ClassificationResults)
{
	cv::Mat ReshapedFeatures;
	ReshapeFeatures(FeatureVectors, ReshapedFeatures);
	cv::Mat Results;
	this->svm.predict(ReshapedFeatures, Results);
	ReshapeLabels(Results, ClassificationResults);
	std::cout << ClassificationResults.size() << std::endl;
}

void DecisionMaker::ReshapeLabels(std::vector<int> &Labels, cv::Mat &ReshapedLabels)
{
	ReshapedLabels = cv::Mat(Labels.size(), 1, CV_32FC1);
	for (int iter = 0; iter < int(Labels.size()); ++iter)
	{
		ReshapedLabels.at<float>(iter) = Labels.at(iter);
		//std::cout << ReshapedLabels.at<float>(iter) << std::endl;
	}

}

void DecisionMaker::ReshapeLabels(cv::Mat &Labels, std::vector<int> &ReshapedLabels)
{
	ReshapedLabels.clear();

	for (int row = 0; row < Labels.rows; ++row)
	{
		for (int col = 0; col < Labels.cols; ++col)
		{
			ReshapedLabels.push_back(int(Labels.at<float>(row, col)));
			//std::cout << Labels.at<float>(row, col) << std::endl;
		}
	}
}


void DecisionMaker::ReshapeFeatures(std::vector<std::vector< cv::Mat >> &FeatureVectors, cv::Mat &ReshapedFeatures)
{
	int Dim = FeatureVectors[0][0].rows*FeatureVectors[0][0].cols;
	ReshapedFeatures = cv::Mat(FeatureVectors.size(), Dim, CV_32FC1);
	int LabelIndex = 0;

	//reshape features to one Mat for the SVM
	for (unsigned int iterClasses = 0; iterClasses < FeatureVectors.size(); ++iterClasses)
	{
		std::vector<cv::Mat>* classFeatures = &FeatureVectors[iterClasses];
		for (unsigned int iterFeatures = 0; iterFeatures < classFeatures->size(); iterFeatures)
		{
			cv::Mat* feature = &classFeatures->operator[](iterFeatures);
			int Index = 0;
			
			for (int i = 0; i < feature->rows; i++)
			{
				for (int j = 0; j < feature->cols; j++)
				{
					ReshapedFeatures.at<float>(LabelIndex, Index++) = feature->at<int>(i, j);
				}
			}
		}
	}
}


//// define the parameters for training the random forest (trees)
//
//float priors[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };  // weights of each classification for classes
//// (all equal as equal samples of each digit)
//
//CvRTParams params = CvRTParams(25, // max depth
//	5, // min sample count
//	0, // regression accuracy: N/A here
//	false, // compute surrogate split, no missing data
//	15, // max number of categories (use sub-optimal algorithm for larger numbers)
//	priors, // the array of priors
//	false,  // calculate variable importance
//	4,       // number of variables randomly selected at node and used to find the best split(s).
//	100,	 // max number of trees in the forest
//	0.01f,				// forrest accuracy
//	CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
//	);
//
//// train random forest classifier (using training data)
//
//printf("\nUsing training database: %s\n\n", argv[1]);
//CvRTrees* rtree = new CvRTrees;
//
//rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
//	Mat(), Mat(), var_type, Mat(), params);