#include "Matrix.h"
 
Matrix::Matrix(){}
 
Matrix::Matrix(int height_, int width_) {
	height = height_;
	width = width_;
	data = std::vector<std::vector<double>> (height_, std::vector<double>(width_));
}
 
Matrix::Matrix(std::vector<std::vector<double>> const &data) {
	this->height = data.size();
	this->width = data[0].size();
	this->data = data;
}
 
Matrix Matrix::multiply(double const &value) {
	Matrix result(height, width);
	int i, j;
    
	for(i = 0; i < height; i++) {
		for(j = 0; j < width; j++) {
			result.data[i][j] = data[i][j] * value;
		}
	}
 
	return result;
}
 
Matrix Matrix::add(Matrix const &m) const {
	Matrix result(height, width);
	int i, j;
 
	for(i = 0; i < height; i++) {
		for(j = 0; j < width; j++) {
			result.data[i][j] = data[i][j] + m.data[i][j];
		}
	}
 
	return result;
}
 
Matrix Matrix::subtract(Matrix const &m) const {
	Matrix result(height, width);
	int i, j;
 
	for(i = 0 ; i < height; i++) {
		for(j = 0; j < width; j++) {
			result.data[i][j] = data[i][j] - m.data[i][j];
		}
	}
 
	return result;
}
 
Matrix Matrix::multiply(Matrix const &m) const {
	Matrix result(height, width);
	int i, j;
 
	for(i = 0; i < height; i++) {
		for(j = 0; j < width; j++) {
			result.data[i][j] = data[i][j] * m.data[i][j];
		}
	}
	
    return result;
}
 
Matrix Matrix::dot(Matrix const &m) const {
	int i, j, h, mwidth = m.width;
	double w = 0;
 
	Matrix result(height, mwidth);
 
	for(i = 0; i < height; i++) {
		for(j = 0; j < mwidth; j++) {
			for(h = 0; h < width ; h++) {
				w += data[i][h]*m.data[h][j];
			}
			result.data[i][j] = w;
			w = 0;
		}
	}
	
    return result;
}
 
Matrix Matrix::transpose() const {
	Matrix result(width, height);
	int i, j;
 
	for(i = 0; i < width; i++){
		for(j = 0; j < height; j++){
			result.data[i][j] = data[j][i];
		}
	}
	
	return result;
}
 
Matrix Matrix::applyFunction(double (*function)(double)) const {
	Matrix result(height, width);
	int i, j;
 
	for(i = 0; i < height; i++) {
		for(j = 0; j < width; j++) {
			result.data[i][j] = function(data[i][j]);
		}
	}
 
	return result;
}