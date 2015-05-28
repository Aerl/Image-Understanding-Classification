#include "stdafx.h"
#include "EvaluationUnit.h"


EvaluationUnit::EvaluationUnit()
{
}


EvaluationUnit::EvaluationUnit(std::vector<Image> TestImages)
{
	this->TestImages = TestImages;
}

double EvaluationUnit::EvaluateResultSimple(std::vector<std::string> &Result)
{
	assert(this->TestImages.size() == Result.size());

	double correctClassifications = 0;
	for (unsigned int iter = 0; iter < this->TestImages.size(); ++iter)
	{
		std::string* OriginalClass = &this->TestImages[iter].category;
		std::string* ResultClass = &Result[iter];

		if (*OriginalClass == *ResultClass)
		{
			correctClassifications++;
		}
	}
	return correctClassifications / double(Result.size());

}
