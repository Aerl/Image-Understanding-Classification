#include "stdafx.h"
#include "ImageLoader.h"

ImageLoader::ImageLoader()
{
}

ImageLoader::ImageLoader(std::string path)
{
	this->path = path;
}

void ImageLoader::LoadAllImagesFromSubfolder(std::string subfolder)
{
	std::string folder = this->path + "/" + subfolder;
	this->LoadImagesFromFolder(folder, -1);
}

void ImageLoader::LoadImagesFromFolder(std::string folder, int maxNumImg)
{
	cout << "Folder: " + folder << endl;
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

					std::string FulImagePath = path + "/" + imgName;
					cv::Mat img;
					img = cv::imread(FulImagePath, cv::IMREAD_COLOR);

					if (!img.data)
					{
						cout << "Invalid Image -  No Data" << endl;
						return;
					}
					images.push_back(img);
				}
			}
		}
		closedir(directory);
	}
}


