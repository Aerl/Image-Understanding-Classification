#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include<string>
#include<vector>

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
	std::vector<cv::Mat> images;
	Parameters parameters;
public:	
	ImageLoader();
	ImageLoader(std::string &path);	
	void LoadAllImagesFromSubfolder(std::string &subfolder);
	void LoadAllImagesFromSubfolders(std::vector<std::string> &subfolders);
	void LoadAllImages();
	void ScaleAndCropImage(cv::Mat &InputImage, cv::Mat &OutpuImage);

	//getters
	std::vector<cv::Mat> getImages();
private:
	void LoadImagesFromFolder(std::string &Folder, int maxNumImg);
};

