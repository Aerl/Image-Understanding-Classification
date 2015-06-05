#pragma once

#include "Image.h"
#include <assert.h> 
#include <string>
#include <vector>

class EvaluationUnit
{
	std::vector<int> Labels;
	int NumberOfClasses;
	int NumberOfSamples;

public:
	EvaluationUnit();
	EvaluationUnit(std::vector<int> Labels, int NumberOfClasses, int NumberOfSamples);
	double EvaluateResultSimple(std::vector<int> &Result);
	void EvaluateResultComplex(std::vector<int> &Result, std::vector<int> &ClassPercentage, std::vector<std::vector<int>> &Statistics);
};

