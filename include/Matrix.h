#pragma once

#include <vector>
#include <sstream>
#include <iostream>

class Matrix {
public:
	Matrix();
	Matrix(int height, int width);
	Matrix(std::vector<std::vector<double>> const &data);
 
	Matrix multiply(double const &value);
 
	Matrix add(Matrix const &m) const;
	Matrix subtract(Matrix const &m) const;
	Matrix multiply(Matrix const &m) const;
 
	Matrix dot(Matrix const &m) const;
	Matrix transpose() const;
 
	Matrix applyFunction(double (*function)(double)) const;
 

	std::vector<std::vector<double>> data;
	int height;
	int width;
};
