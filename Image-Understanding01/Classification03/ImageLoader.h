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
	std::string path;
	std::vector<cv::Mat> images;
	std::vector<std::string> folders;
public:
	ImageLoader();
	ImageLoader(std::string path);	
	void LoadAllImagesFromSubfolder(std::string subfolder);
	void LoadAllImagesFromSubfolders(std::vector<std::string> subfolders);
	void LoadAllImages();
private:
	void LoadImagesFromFolder(std::string Folder, int maxNumImg);
};

