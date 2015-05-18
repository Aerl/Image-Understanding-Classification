#include "stdafx.h"
#include "ImageLoader.h"

ImageLoader::Parameters::Parameters()
{
	this->ROIx = 300;
	this->ROIy = 300;
}

ImageLoader::ImageLoader()
{
}

ImageLoader::ImageLoader(std::string &path)
{
	//Contructor with path to the '101_ObjectCategories'-folder
	this->path = path;
}

std::vector<cv::Mat> ImageLoader::getImages()
{
	return this->images;
}

void ImageLoader::LoadAllImagesFromSubfolder(std::string &subfolder)
{
	//Load all images from a single subfolder of '101_ObjectCategories'
	std::string folder = this->path + "/" + subfolder;
	this->LoadImagesFromFolder(folder, -1);
}

void ImageLoader::LoadImagesFromSubfolder(std::string &subfolder, int maxNumImg)
{
	//Load a certain number of images from a single subfolder of '101_ObjectCategories'
	std::string folder = this->path + "/" + subfolder;
	this->LoadImagesFromFolder(folder, maxNumImg);
}

void ImageLoader::LoadAllImagesFromSubfolders(std::vector<std::string> &subfolders)
{
	//Load all images from a list of subfolders of '101_ObjectCategories'
	for (std::vector<std::string>::iterator iter = subfolders.begin(); iter != subfolders.end(); ++iter)
	{
		std::string folder = this->path + "/" + *iter;
		this->LoadImagesFromFolder(folder, -1);
	}
}

void ImageLoader::LoadImagesFromSubfolders(std::vector<std::string> &subfolders, int maxNumImg)
{
	//Load a certain number of images from a list of subfolders of '101_ObjectCategories'
	for (std::vector<std::string>::iterator iter = subfolders.begin(); iter != subfolders.end(); ++iter)
	{
		std::string folder = this->path + "/" + *iter;
		this->LoadImagesFromFolder(folder, maxNumImg);
	}
}

void ImageLoader::LoadAllImages()
{
	//Load all images from all subfolder
	//this will probably lead to a memory error
	DIR* directory = opendir(this->path.c_str());
	struct dirent *subfolder;
	if (directory != NULL)
	{
		while (subfolder = readdir(directory))
		{
			if (subfolder != NULL)
			{
				if (subfolder->d_type == 16384) //type is folder
				{
					if (subfolder->d_name != "." && subfolder->d_name != "..")
					{
						LoadImagesFromFolder(this->path + "/" + subfolder->d_name, -1);
					}
				}
			}
		}
	}
}

void ImageLoader::LoadImages(int maxNumImg)
{
	//Load a certain number of images from all subfolders
	DIR* directory = opendir(this->path.c_str());
	struct dirent *subfolder;
	if (directory != NULL)
	{
		while (subfolder = readdir(directory))
		{
			if (subfolder != NULL)
			{
				if (subfolder->d_type == 16384) //type is folder
				{
					if (subfolder->d_name != "." && subfolder->d_name != "..")
					{
						LoadImagesFromFolder(this->path + "/" + subfolder->d_name, maxNumImg);
					}
				}
			}
		}
	}
}

void ImageLoader::LoadImagesFromFolder(std::string &folder, int maxNumImg)
{
	//Load images from the given directory
	//TODO: Implement random selection of restricted number of entries.
	//-1 -> all images from folder
	std::cout << "Folder: " + folder << std::endl;
	std::vector<cv::Mat> CurrentFolder;
	DIR* directory = opendir(folder.c_str());
	std::string imgName;
	struct dirent *entry;

	if (directory != NULL)
	{
		while (entry = readdir(directory))
		{
			if (entry != NULL)
			{
				imgName = entry->d_name;
				if (imgName != "." && imgName != "..")
				{
					std::string FullImagePath = folder + "/" + imgName;
					//std::cout << "Full Path: " + FullImagePath << std::endl;
					cv::Mat img;
					img = cv::imread(FullImagePath);
					if (!img.data) //check if image is not empty
					{
						std::cout << "Invalid Image -  No Data" << std::endl;
						return;
					}
					cv::Mat out;
					ScaleAndCropImage(img, out);
					CurrentFolder.push_back(out);

				}
			}
		}
		closedir(directory);
		SelectAndCopyImages(CurrentFolder, maxNumImg);
		std::cout << "   Fully Loaded" << std::endl;
	}
}

void ImageLoader::ScaleAndCropImage(cv::Mat &InputImage, cv::Mat &OutpuImage)
{
	//Image is first resized, so that the smaller side is 300 Pixels
	//Region of intrest is then cut out in the middle according to parameters.ROIx

	double ScalingFactor = double(max(this->parameters.ROIx, this->parameters.ROIy)) / double(min(InputImage.rows, InputImage.cols));
	//std::cout << "  Image: " + std::to_string(InputImage.rows) + " x " + std::to_string(InputImage.cols) << std::endl;
	//std::cout << "  ScalingFactor: " + std::to_string(ScalingFactor) << std::endl;
	cv::resize(InputImage, OutpuImage, cv::Size(), ScalingFactor, ScalingFactor, cv::INTER_LINEAR);

	int xDifference = abs(OutpuImage.cols - this->parameters.ROIx) / 2;
	int yDifference = abs(OutpuImage.rows - this->parameters.ROIy) / 2;

	//std::cout << "  ROI Position: " + std::to_string(xDifference) + " x " + std::to_string(yDifference) << std::endl;
	//std::cout << "  ROI Size: " + std::to_string(this->parameters.ROIx) + " x " + std::to_string(this->parameters.ROIy) << std::endl;

	cv::Rect newROI(xDifference, yDifference, this->parameters.ROIx, this->parameters.ROIy);
	OutpuImage = OutpuImage(newROI);
}

void ImageLoader::SelectAndCopyImages(std::vector<cv::Mat> &AllImages, int maxNumImg)
{
	//According to maxNumImg random images are selected from the folder.
	//If maxNumImg is larger than the number of elements in the folder, all images are loaded instead.

	//std::cout << "  Maximum Number Of Images: " + std::to_string(maxNumImg) << std::endl;
	//std::cout << "  Images in Folder: " + std::to_string(AllImages.size()) << std::endl;

	if (maxNumImg < 0)
	{
		CopyAllImages(AllImages);
	}
	else
	{
		int NumEl = AllImages.size();
		if (maxNumImg >= NumEl)
		{
			CopyAllImages(AllImages);
		}
		else
		{
			std::vector<int> indices;
			srand(time(NULL));
			for (int iter = 0; iter < maxNumImg; ++iter)
			{
				int index = rand() % NumEl;
				if (std::find(indices.begin(), indices.end(), index) != indices.end())
				{
					iter--;
				}
				else
				{
					//std::cout << "  Index: " + std::to_string(index) << std::endl;
					this->images.push_back(AllImages[index]);
					indices.push_back(index);
				}
			}
		}
	}
}

void ImageLoader::CopyAllImages(std::vector<cv::Mat> &AllImages)
{
	//All Images from the subfolder are copied into the images vector
	for (std::vector<cv::Mat>::iterator iter = AllImages.begin(); iter != AllImages.end(); ++iter)
	{
		this->images.push_back(*iter);
	}
}


