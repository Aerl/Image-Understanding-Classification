#pragma once

#include "Image.h"
#include <assert.h> 
#include <string>
#include <vector>

class EvaluationUnit
{
	std::vector<Image> TestImages;

public:
	EvaluationUnit();
	EvaluationUnit(std::vector<Image> TestImages);
	double EvaluateResultSimple(std::vector<std::string> &Result);
};

