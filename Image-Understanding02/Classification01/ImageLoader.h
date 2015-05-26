#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "Image.h"
#include <string>
#include <vector>

class ImageLoader
{
	struct Parameters
	{
		int ROIx;
		int ROIy;
		//! Default constructor ensuring that all variables are initialized.
		Parameters();
	};

	std::string path;	
	std::vector<Image> TrainingImages;
	std::vector<Image> TestImages;
	Parameters parameters;
public:	
	//constructor
	ImageLoader();
	ImageLoader(std::string &path);	
	//functions
	void LoadImagesFromSubfolders(std::vector<std::string> &subfolders);
	void LoadImages();
	//getter
	std::vector<Image> getTrainingImages();
	std::vector<Image> getTestImages();
private:
	int getNumberOfImages(std::string &Folder);
	void ScaleAndCropImage(cv::Mat &InputImage, cv::Mat &OutpuImage);
	void LoadImagesFromFolder(std::string &Folder, int NumImg);
	void SelectAndCopyImages(std::vector<cv::Mat> &AllImages, int NumImg, std::string category);
};

