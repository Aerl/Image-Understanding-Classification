#pragma once

#include "Image.h"
#include <assert.h> 
#include <string>
#include <vector>

class EvaluationUnit
{
	std::vector<int> TestLabels;

public:
	EvaluationUnit();
	EvaluationUnit(std::vector<int> TestLabels);
	double EvaluateResultSimple(std::vector<int> &Result);
};

