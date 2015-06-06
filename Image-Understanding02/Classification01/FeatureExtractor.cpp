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

void FeatureExtractor::computeSURFFeatures(std::vector<cv::Mat> &TrainImages, std::vector<int> trainingLabels, std::vector<cv::Mat> &FeatureVectorsSURFUnclustered)
{
	int minHessian = 400;

	cv::SurfFeatureDetector detector(minHessian, 4, 2, false);
	cv::SurfDescriptorExtractor surf;

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptor;

	for (int i = 0; i < int(TrainImages.size()); i++)
	{
		detector.detect(TrainImages[i], keypoints);
		detector.compute(TrainImages[i], keypoints, descriptor);
		int pushIndex = trainingLabels[i];
		FeatureVectorsSURFUnclustered[pushIndex].push_back(descriptor);
	}
}

void FeatureExtractor::getBagOfWords(std::vector<cv::Mat> &TestImages, std::vector<cv::Mat> &FeatureVectorsSURFUnclustered, cv::Mat &dictionary, std::vector<cv::Mat> &clusteredFeatures)
{
	int minHessian = 400;
	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize = 200;
	//define Term Criteria
	cv::TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	//retries number
	int retries = 1;
	//necessary flags
	int flags = cv::KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	cv::BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	
	//cluster the feature vectors
	for (int i = 0; i < int(FeatureVectorsSURFUnclustered.size()); i++)
	{
		dictionary = bowTrainer.cluster(FeatureVectorsSURFUnclustered[i]);

		////store the vocabulary
		//cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
		//fs << "vocabulary" << dictionary;
		//fs.release();

		//create a nearest neighbor matcher
		cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);
		//create SURF feature point extracter
		cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(minHessian, 4, 2, false));
		//create SURF descriptor extractor
		cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor(minHessian, 4, 2, false));
		//create BoF (or BoW) descriptor extractor
		cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
		//Set the dictionary with the vocabulary we created in the first step
		bowDE.setVocabulary(dictionary);

			////To store the image file name
			//char * filename = new char[100];
			////To store the image tag name - only for save the descriptor in a file
			//char * imageTag = new char[10];
			////open the file to write the resultant descriptor
			//cv::FileStorage fs1("descriptor.yml", cv::FileStorage::WRITE);
			////the image file with the location. change it according to your image file location
			//sprintf(filename, "G:\\testimages\\image\\1.jpg");
			////read the image
			//cv::Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		for (int j = 0; j < int(TestImages.size()); j++)
			{
				//To store the keypoints that will be extracted by SURF
				std::vector<cv::KeyPoint> keypoints;
				//Detect SURF keypoints (or feature points)
				detector->detect(TestImages[j], keypoints);
				//To store the BoW (or BoF) representation of the image
				cv::Mat bowDescriptor;
				//extract BoW (or BoF) descriptor from given image
				bowDE.compute(TestImages[j], keypoints, bowDescriptor);
				clusteredFeatures[j].push_back(bowDescriptor);
			}
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
	hog.blockSize = cv::Size(8,8);
	hog.cellSize = cv::Size(8,8);
	hog.nbins = 4;
	for (int iter = 0; iter < Size; ++iter)
	{
		cv::cvtColor(Images[iter], gray, cv::COLOR_BGR2GRAY);
		
		hog.compute(gray, descriptors, cv::Size(hog.winSize.width/2, hog.winSize.height/2), cv::Size(0, 0), location);
		std::cout << "  Feature Size: " + std::to_string(descriptors.size()) << std::endl;
		cv::Mat help = cv::Mat(descriptors).clone();
		FeatureVectors[iter].push_back(help);

		/// Display
		//cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		//imshow("calcHist Demo", help);

		//cv::waitKey(0);
	}
}

void FeatureExtractor::computeColorFeatures(std::vector<cv::Mat> &Images, std::vector<std::vector< cv::Mat >> &FeatureVectorsColor)
{
	for (int iter = 0; iter < int(Images.size()); ++iter)
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
		
		//std::cout << "  b_hist: " + std::to_string(b_hist.rows) + " / " + std::to_string(b_hist.cols) << std::endl;
		//std::cout << "  g_hist: " + std::to_string(g_hist.rows) + " / " + std::to_string(g_hist.cols) << std::endl;
		//std::cout << "  r_hist: " + std::to_string(r_hist.rows) + " / " + std::to_string(r_hist.cols) << std::endl;

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


