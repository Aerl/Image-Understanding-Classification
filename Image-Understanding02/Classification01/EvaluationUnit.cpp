#include "stdafx.h"
#include "EvaluationUnit.h"


EvaluationUnit::EvaluationUnit()
{
}


EvaluationUnit::EvaluationUnit(std::vector<int> Labels, int NumberOfClasses, int NumberOfSamples)
{
	this->Labels = Labels;
	this->NumberOfClasses = NumberOfClasses;
	this->NumberOfSamples = NumberOfSamples;
}

double EvaluationUnit::EvaluateResultSimple(std::vector<int> &Result)
{
	//returns Percentage of correctly classified images
	assert(this->Labels.size() == Result.size());

	double correctClassifications = 0;
	for (unsigned int iter = 0; iter < this->Labels.size(); ++iter)
	{

		if (this->Labels[iter] == Result[iter])
		{
			correctClassifications++;
		}
	}
	return correctClassifications / double(Result.size());

}

void EvaluationUnit::EvaluateResultComplex(std::vector<int> &Result, std::vector<double> &ClassPercentage, std::vector<std::vector<int>> &Statistics)
{
	//returns classification statistics: as which class were items from each class classified
	
	ClassPercentage.clear();
	ClassPercentage.resize(this->NumberOfClasses);
	std::fill(ClassPercentage.begin(), ClassPercentage.end(), 0);

	//make sure statistics has size NumberOfClasses*NumberOfClasses and each entry is 0
	Statistics.clear();
	Statistics.resize(this->NumberOfClasses);
	for (std::vector<std::vector<int>>::iterator iter = Statistics.begin(); iter != Statistics.end(); ++iter)
	{
		iter->clear();
		iter->resize(this->NumberOfClasses);
		std::fill(iter->begin(), iter->end(), 0);
	}
	
	//Iterate over all Samples
	for (unsigned int iter = 0; iter < this->Labels.size(); ++iter)
	{
		std::vector<int>* ClassStats = &Statistics[this->Labels[iter]];
		assert(this->Labels.size() == Result.size());
		ClassStats->operator[](Result[iter])++;		

		if (this->Labels[iter] == Result[iter])
		{
			ClassPercentage[Labels[iter]]++;
		}
	}

	//compute percentages
	for (std::vector<double>::iterator iter = ClassPercentage.begin(); iter != ClassPercentage.end(); ++iter)
	{
		*iter = *iter / double(this->NumberOfSamples);
	}

}
