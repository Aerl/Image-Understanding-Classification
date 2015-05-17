// Classification02.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>


using namespace cv;
using namespace std;


int _tmain(int argc, _TCHAR* argv[])
{
	vector<Mat> images;
	vector<string> folders;
	folders.push_back("../101_ObjectCategories/accordion");
	folders.push_back("../101_ObjectCategories/crab");
	folders.push_back("../101_ObjectCategories/garfield");
	folders.push_back("../101_ObjectCategories/octopus");
	folders.push_back("../101_ObjectCategories/scissors");
	folders.push_back("../101_ObjectCategories/sunflower");
	folders.push_back("../101_ObjectCategories/wrench");

	for (vector<string>::iterator iter = folders.begin(); iter != folders.end(); ++iter)
	{
		string path = *iter;

		//string path("../101_ObjectCategories/accordion");
		//string path("D:/GitHub/Image-Understanding-Classification/Image-Understanding01/101_ObjectCategories/accordion");
		//string path("D:/HELP");
		namedWindow("Show Images", WINDOW_AUTOSIZE);
		cout << "Image Path: " + path << endl;


		DIR* directory = opendir(path.c_str());
		string imgName;
		struct dirent *entry;

		if (directory != NULL) {
			while (entry = readdir(directory)) {
				if (entry != NULL)
				{
					imgName = entry->d_name;

					if (imgName != "." && imgName != "..")
					{
						cout << "Image Name: " + imgName << endl;

						string ImageLocation = path + "/" + imgName;

						Mat img;
						//img = imread("D:/HELP/image_01.jpg", IMREAD_COLOR);
						img = imread(ImageLocation, IMREAD_COLOR);

						if (!img.data)
						{
							//cout << directory << endl;						
							cout << "Invalid Image -  data is empty" << endl;
							return -1;
						}

						imshow("Show Images", img);
						images.push_back(img);
						waitKey(100);
					}
				}
			}
			closedir(directory);
		}
		else {
			cout << "not present" << endl;
		}
	}

	cout << "Number of Images: " + images.size() << endl;
	
	return 0;
}

