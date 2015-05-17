#include "stdafx.h"
#include "ImageLoader.h"

ImageLoader::ImageLoader()
{
}

ImageLoader::ImageLoader(std::string path)
{
	//Contructor with path to the '101_ObjectCategories'-folder
	this->path = path;
}

void ImageLoader::LoadAllImagesFromSubfolder(std::string subfolder)
{
	//Load all images from a single subfolder of '101_ObjectCategories'
	std::string folder = this->path + "/" + subfolder;
	this->LoadImagesFromFolder(folder, -1);
}

void ImageLoader::LoadAllImagesFromSubfolders(std::vector<std::string> subfolders)
{
	//Load all images from a list of subfolders of '101_ObjectCategories'
	for (std::vector<std::string>::iterator iter = subfolders.begin(); iter != subfolders.end(); ++iter)
	{
		std::string subfolder = *iter;
		std::string folder = this->path + "/" + subfolder;
		this->LoadImagesFromFolder(folder, -1);
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
					LoadImagesFromFolder(this->path + "/" + subfolder->d_name, -1);
				}
			}
		}
	}
}

void ImageLoader::LoadImagesFromFolder(std::string folder, int maxNumImg)
{
	//Load images from the given directory
	//TODO: Implement random selection of restricted number of entries.
	std::cout << "Folder: " + folder << std::endl;

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
					this->images.push_back(img);
				}
			}
		}
		closedir(directory);
		std::cout << "   Fully Loaded" << std::endl;
	}
}


