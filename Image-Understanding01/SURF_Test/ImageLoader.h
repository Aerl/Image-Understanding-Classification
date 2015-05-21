#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <stdlib.h>
#include <time.h>
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
	std::vector<cv::Mat> images;
	Parameters parameters;
public:
	//constructor
	ImageLoader();
	ImageLoader(std::string &path);
	//functions
	void LoadAllImagesFromSubfolder(std::string &subfolder);
	void LoadImagesFromSubfolder(std::string &subfolder, int maxNumImg);
	void LoadAllImagesFromSubfolders(std::vector<std::string> &subfolders);
	void LoadImagesFromSubfolders(std::vector<std::string> &subfolders, int maxNumImg);
	void LoadAllImages();
	void LoadImages(int maxNumImg);
	//getter
	std::vector<cv::Mat> getImages();
private:
	void ScaleAndCropImage(cv::Mat &InputImage, cv::Mat &OutpuImage);
	void LoadImagesFromFolder(std::string &Folder, int maxNumImg);
	void SelectAndCopyImages(std::vector<cv::Mat> &AllImages, int maxNumImg);
	void ImageLoader::CopyAllImages(std::vector<cv::Mat> &AllImages);
};

