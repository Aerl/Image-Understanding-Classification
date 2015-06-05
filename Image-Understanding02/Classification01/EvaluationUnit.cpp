#include "stdafx.h"
#include "EvaluationUnit.h"


EvaluationUnit::EvaluationUnit()
{
}


EvaluationUnit::EvaluationUnit(std::vector<int> TestLabels)
{
	this->TestLabels = TestLabels;
}

double EvaluationUnit::EvaluateResultSimple(std::vector<int> &Result)
{
	assert(this->TestLabels.size() == Result.size());

	double correctClassifications = 0;
	for (unsigned int iter = 0; iter < this->TestLabels.size(); ++iter)
	{

		if (this->TestLabels[iter] == Result[iter])
		{
			correctClassifications++;
		}
	}
	return correctClassifications / double(Result.size());

}
