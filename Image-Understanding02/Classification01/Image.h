#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
class Image
{
public:
	cv::Mat data;
	std::string category;

	Image();
	Image(cv::Mat data, std::string category);
};

