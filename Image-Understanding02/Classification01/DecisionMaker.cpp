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

//----------------SVMs------------------

void DecisionMaker::TrainSVM(cv::Mat &FeatureVectors, std::vector<int> &Labels)
{
	cv::Mat ReshapedLabels;
	ReshapeLabels(Labels, ReshapedLabels);

	// set up SVM parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 500, 1e-6);

	// set up Support Vector Machine for training and classification 
	this->svm.train(FeatureVectors, ReshapedLabels, cv::Mat(), cv::Mat(), params);
}

void DecisionMaker::PredictSVM(cv::Mat &FeatureVectors, std::vector<int> &ClassificationResults)
{
	cv::Mat Results;
	this->svm.predict(FeatureVectors, Results);
	ReshapeLabels(Results, ClassificationResults);
	//std::cout << ClassificationResults.size() << std::endl;
}

//----------------Random Trees------------------

void DecisionMaker::TrainRandomTrees(cv::Mat &FeatureVectors, std::vector<int> &Labels)
{
	CvRTParams params = CvRTParams();

	params.max_depth = 60;
	params.min_sample_count = FeatureVectors.rows / 100; //1%
	params.max_categories = 100;

	/*(25, // max depth
	5, // min sample count
	0, // regression accuracy: N/A here
	false, // compute surrogate split, no missing data
	15, // max number of categories (use sub-optimal algorithm for larger numbers)
	priors, // the array of priors
	false,  // calculate variable importance
	4,       // number of variables randomly selected at node and used to find the best split(s).
	100,	 // max number of trees in the forest
	0.01f,				// forrest accuracy
	CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
	);*/

	cv::Mat ReshapedLabels;
	ReshapeLabels(Labels, ReshapedLabels);

	this->rtree.train(FeatureVectors, CV_ROW_SAMPLE, ReshapedLabels, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
}

void DecisionMaker::PredictRandomTrees(cv::Mat &FeatureVectors, std::vector<int> &ClassificationResults)
{
	ClassificationResults = std::vector<int>(FeatureVectors.rows);

	for (int iter = 0; iter < FeatureVectors.rows; ++iter)
	{
		cv::Mat Sample = FeatureVectors.row(iter);
		/*		cv::Mat FloatSample;
		Sample.convertTo(FloatSample, CV_32FC1); */
		float SampleResult = this->rtree.predict(Sample);
		//std::cout << SampleResult << std::endl;
		ClassificationResults[iter] = int(SampleResult);
	}

}

//----------------Feature Reduction------------------

void DecisionMaker::constructPCA(cv::Mat &Features)
{
	pca = cv::PCA(Features, // pass the data
		cv::Mat(), // we do not have a pre-computed mean vector,
		// so let the PCA engine to compute it
		0, // DATA_AS_ROW indicate that the vectors
		// are stored as matrix rows
		// (use PCA::DATA_AS_COL if the vectors are
		// the matrix columns)
		Features.cols / 2 // specify, how many principal components to retain
		);
	std::cout << "PCA constructed" << std::endl;
}

void DecisionMaker::reduceFeaturesPCA(cv::Mat &Features, cv::Mat &ReducedFeatures)
{
	pca.project(Features, ReducedFeatures);

	std::cout << "Features: " + std::to_string(Features.cols) << std::endl;
	std::cout << "ReducedFeatures: " + std::to_string(ReducedFeatures.cols) << std::endl;
}

//----------------Reshaping------------------

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
	int num_files = FeatureVectors.size();
	int Dim = 0;

	for (std::vector<cv::Mat>::iterator iter = FeatureVectors[0].begin(); iter != FeatureVectors[0].end(); ++iter)
	{
		Dim += iter->rows*iter->cols;
	}

	//std::cout << "Dim: " + std::to_string(Dim) << std::endl;

	ReshapedFeatures = cv::Mat(num_files, Dim, CV_32FC1);

	// reshape Images to one Mat for the SVM
	int labelIndex = 0;
	for (std::vector<cv::Mat> fvector : FeatureVectors)
	{
		int ii = 0;

		for (cv::Mat img_mat : fvector)
		{

			for (int i = 0; i < img_mat.rows; i++)
			{
				for (int j = 0; j < img_mat.cols; j++)
				{
					ReshapedFeatures.at<float>(labelIndex, ii++) = img_mat.at<int>(i, j);

				}
			}
		}
		labelIndex++;
	}
}