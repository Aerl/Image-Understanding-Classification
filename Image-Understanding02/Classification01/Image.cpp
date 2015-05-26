#include "stdafx.h"
#include "Image.h"


Image::Image()
{
}


Image::Image(cv::Mat data, std::string category)
{
	this->data = data;
	this->category = category;
}
