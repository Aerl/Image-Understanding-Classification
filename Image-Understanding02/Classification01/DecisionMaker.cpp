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

void DecisionMaker::MakeDecisionSVM(std::vector<std::vector< cv::Mat >> &features, std::vector<int> &trainingLabels, std::vector<int> &classificationResults)
{

	//int num_files = features.size();
	//int img_area = 150 * 150;
	//cv::Mat labels(num_files, 1, CV_32FC1);
	//cv::Mat training_mat(num_files, img_area, CV_32FC1);
	//int labelIndex = 0;

	//// reshape features to one Mat for the SVM
	//for (std::vector< cv::Mat > feature : features)
	//{
	//	int ii = 0;
	//	for (int i = 0; i < feature.rows; i++)
	//	{
	//		for (int j = 0; j < feature.cols; j++)
	//		{
	//			training_mat.at<float>(labelIndex, ii++) = feature.at<uchar>(i, j);
	//		}
	//	}
	//	labels.at<float>(labelIndex) = trainingLabels.at(labelIndex);
	//	labelIndex++;
	//}

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
	//cv::Mat test2 = training_mat.row(18);

	//int p = svm.predict(test2);
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