#include "stdafx.h"
#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{

}

FeatureExtractor::Parameters::Parameters()
{
	this->WindowX = 144;
	this->WindowY = 144;
	this->histSize = 256;
}

FeatureExtractor::~FeatureExtractor()
{

}

void FeatureExtractor::computeSURFFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectorsSURF)
{

	int minHessian = 400;

	cv::SurfFeatureDetector detector(minHessian);
	cv::SurfDescriptorExtractor surf;

	std::vector<std::vector<cv::KeyPoint>> keypoints;
	cv::Mat descriptor;
	int index = 0;

	for (cv::Mat image : Images)
	{
		keypoints.push_back(std::vector<cv::KeyPoint>());
		detector.detect(image, keypoints.at(index));
		surf.compute(image, keypoints.at(index), descriptor);
		FeatureVectorsSURF[index].push_back(descriptor.clone());
		index++;
	}

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
		//std::cout << "  Feature Size: " + std::to_string(descriptors.size()) << std::endl;
		FeatureVectors[iter].push_back(cv::Mat(descriptors).clone());
	}
}

void FeatureExtractor::computeColorFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectorsColor)
{
	for (int iter = 0; iter < Images.size(); ++iter)
	{
		std::vector<cv::Mat> bgr_planes;
		split(Images[iter], bgr_planes);

		/// Set the ranges ( for B,G,R) )
		float range[] = { 0, this->parameters.histSize };
		const float* histRange = { range };

		bool uniform = true; 
		bool accumulate = false;

		cv::Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &this->parameters.histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &this->parameters.histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &this->parameters.histSize, &histRange, uniform, accumulate);

		FeatureVectorsColor[iter].push_back(b_hist);
		FeatureVectorsColor[iter].push_back(g_hist);
		FeatureVectorsColor[iter].push_back(r_hist);

		//// Draw the histograms for B, G and R
		//int hist_w = 512; int hist_h = 400;
		//int bin_w = cvRound((double)hist_w / this->parameters.histSize);

		//cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

		///// Normalize the result to [ 0, histImage.rows ]
		//normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
		//normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
		//normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

		///// Draw for each channel
		//for (int i = 1; i < this->parameters.histSize; i++)
		//{
		//	line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
		//		cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
		//		cv::Scalar(255, 0, 0), 2, 8, 0);
		//	line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
		//		cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
		//		cv::Scalar(0, 255, 0), 2, 8, 0);
		//	line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
		//		cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
		//		cv::Scalar(0, 0, 255), 2, 8, 0);
		//}

		///// Display
		//cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		//imshow("calcHist Demo", histImage);

		//cv::waitKey(0);
	}
}

void FeatureExtractor::MakeDecisionFLANN(std::vector<std::vector< cv::Mat >> &SURFTrain, std::vector<std::vector< cv::Mat >> &SURFTest, std::vector<int> &trainingLabels, std::vector<int> &classificationResults)
{

	cv::FlannBasedMatcher FLANNmatcher;
	cv::Mat featureVectorTest, featureVectorTrain;
	int classIndex = 0;

	for (std::vector< cv::Mat > featureTest : SURFTest)
	{
		std::cout << "Image Number : " + std::to_string(classIndex) << std::endl;
		featureVectorTest = featureTest[0];
		int index = 0;

		std::vector<int> numberGoodMatches(SURFTest.size());

		for (std::vector<cv::Mat> featureTrain : SURFTrain)
		{
			featureVectorTrain = featureTrain[0];
			std::vector<cv::DMatch> matches;
			FLANNmatcher.match(featureVectorTest, featureVectorTrain, matches);
			//std::cout << "Matches Size: " + std::to_string(matches.size()) << std::endl;	
		}	
	}
}


