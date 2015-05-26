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

std::vector<cv::Mat> ImageLoader::getTrainingImages()
{
	return this->TrainingImages;
}
std::vector<cv::Mat> ImageLoader::getTestImages()
{
	return this->TestImages;
}

void ImageLoader::LoadImagesFromSubfolders(std::vector<std::string> &subfolders)
{
	int minNumImg = INT_MAX;

	//Find smallest number of images in folders
	for (std::vector<std::string>::iterator iter = subfolders.begin(); iter != subfolders.end(); ++iter)
	{
		std::string folder = this->path + "/" + *iter;
		std::cout << "Folder: " + *iter << std::endl;
		int SizeOfFolder = this->getNumberOfImages(folder);
		minNumImg = min(SizeOfFolder, minNumImg);
	}

	std::cout << "Minimum Number of Images: " + std::to_string(minNumImg) << std::endl;

	//Load images for training and testing from each subfolder
	for (std::vector<std::string>::iterator iter = subfolders.begin(); iter != subfolders.end(); ++iter)
	{
		std::string folder = this->path + "/" + *iter;
		this->LoadImagesFromFolder(folder, minNumImg);
	}
}


void ImageLoader::LoadImages()
{
	//Find smallest number of images in folders
	int minNumImg = INT_MAX;
	DIR* directory = opendir(this->path.c_str());
	struct dirent *subfolder;
	std::string folderName;
	if (directory != NULL)
	{
		while (subfolder = readdir(directory))
		{
			if (subfolder != NULL)
			{
				if (subfolder->d_type == 16384) //type is folder
				{
					folderName = subfolder->d_name;
						if (folderName != "." && folderName != "..")
					{
						std::string folder = this->path + "/" + folderName;
						std::cout << "Folder: " + folderName << std::endl;
						int SizeOfFolder = this->getNumberOfImages(folder);
						minNumImg = min(SizeOfFolder, minNumImg);
					}
				}
			}
		}
	}
	closedir(directory);
	std::cout << "Minimum Number of Images: " + std::to_string(minNumImg) << std::endl;
	//Load images from all subfolders
	directory = opendir(this->path.c_str());
	if (directory != NULL)
	{
		while (subfolder = readdir(directory))
		{
			if (subfolder != NULL)
			{
				if (subfolder->d_type == 16384) //type is folder
				{
					folderName = subfolder->d_name;
					if (folderName != "." && folderName != "..")
					{
						LoadImagesFromFolder(this->path + "/" + folderName, minNumImg);
					}
				}
			}
		}
	}
	closedir(directory);
}


int ImageLoader::getNumberOfImages(std::string &Folder)
{
	// Returns the number of images in the given folder
	int NumOfImg = 0;
	DIR* directory = opendir(Folder.c_str());
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
					NumOfImg++;
				}
			}
			else
			{
				std::cout << "Invalid Entry - NULL" << std::endl;
			}
		}
		closedir(directory);
	}
	return NumOfImg;
}

void ImageLoader::LoadImagesFromFolder(std::string &folder, int NumImg)
{
	//Load images from the given directory
	//Random selection of restricted number of entries.
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
			else
			{
				std::cout << "Invalid Entry - NULL" << std::endl;
			}
		}
		closedir(directory);
		SelectAndCopyImages(CurrentFolder, NumImg);
		std::cout << "   Fully Loaded" << std::endl;
	}
}

void ImageLoader::ScaleAndCropImage(cv::Mat &InputImage, cv::Mat &OutpuImage)
{
	//Image is first resized, so that the smaller side is 300 Pixels
	//Region of intrest is then cut out in the middle according to parameters.ROIx

	double ScalingFactor = double(max(this->parameters.ROIx, this->parameters.ROIy)) / double(min(InputImage.rows, InputImage.cols));
	cv::resize(InputImage, OutpuImage, cv::Size(), ScalingFactor, ScalingFactor, cv::INTER_LINEAR);

	int xDifference = abs(OutpuImage.cols - this->parameters.ROIx) / 2;
	int yDifference = abs(OutpuImage.rows - this->parameters.ROIy) / 2;

	cv::Rect newROI(xDifference, yDifference, this->parameters.ROIx, this->parameters.ROIy);
	OutpuImage = OutpuImage(newROI);
}

void ImageLoader::SelectAndCopyImages(std::vector<cv::Mat> &AllImages, int NumImg)
{
	//According to maxNumImg random images are selected from the folder.
	//If maxNumImg is larger than the number of elements in the folder, all images are loaded instead.

	//in case of uneven number
	if (NumImg % 2 != 0)
	{
		NumImg--;
	}

	//std::cout << "  Maximum Number Of Images: " + std::to_string(maxNumImg) << std::endl;
	//std::cout << "  Images in Folder: " + std::to_string(AllImages.size()) << std::endl;


	int NumEl = AllImages.size();
	std::vector<int> indices;
	srand(time(NULL));

	//Training Images
	for (int iter = 0; iter < NumImg / 2; ++iter)
	{
		int index = rand() % NumEl;
		if (std::find(indices.begin(), indices.end(), index) != indices.end())
		{
			iter--;
		}
		else
		{
			std::cout << "  Index: " + std::to_string(index) << std::endl;
			this->TrainingImages.push_back(AllImages[index]);
			indices.push_back(index);
		}
	}

	//Test Images
	for (int iter = 0; iter < NumImg / 2; ++iter)
	{
		int index = rand() % NumEl;
		if (std::find(indices.begin(), indices.end(), index) != indices.end())
		{
			iter--;
		}
		else
		{
			std::cout << "  Index: " + std::to_string(index) << std::endl;
			this->TestImages.push_back(AllImages[index]);
			indices.push_back(index);
		}
	}


}



