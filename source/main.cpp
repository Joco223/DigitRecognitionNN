#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "Matrix.h"

Matrix X, W1, H, W2, H2, W3, Y, B1, B2, B3, Y2, dJdB1, dJdB2, dJdB3, dJdW1, dJdW2, dJdW3;
double learningRate;

double random(double x) {
	return (double)(rand() % 10000 + 1)/10000;
}

double sigmoid(double x) {
	return 1/(1+exp(-x));
}

double sigmoidePrime(double x) {
	return exp(-x)/(pow(1+exp(-x), 2));
}

double tanH(double x) {
	return tanh(x);
}

double tanhPrime(double x) {
	return 1 - tanh(x)*tanh(x);
}

double relu(double x) {
	if(x > 0) {
		if(x > 6) {
			return 6;
		}else{
			return x;
		}
	}else{
		return 0;
	}
}

double reluPrime(double x) {
	if(x > 0) {
		return 1;
	}else{
		return 0;
	}
}

std::vector<std::vector<double>> loadImageData(std::string path, int numOfImages) {
	std::ifstream file;
	file.open(path, std::ios::binary);

	std::vector<std::vector<double>> imageData;
	imageData.resize(numOfImages);

	std::vector<char> buffer;
	buffer.resize(numOfImages * 28 * 28 + 16);
	file.read(buffer.data(), numOfImages * 28 * 28 + 16);
	int currentPic = -1;
	bool first = true;

	for(int i = 0; i < numOfImages * 28 * 28; i++) {
		if(i % (28 * 28 + 1) == 0) { currentPic++; }
		//if(buffer[i + 16] != 0) {std::cout << (int)(unsigned char)buffer[i + 16] << '\n'; }
		imageData[currentPic].push_back(buffer[i + 16]);
	}

	for(int i = 0; i < numOfImages; i++) {
		imageData[i].pop_back();
	}

	return imageData;
}

std::vector<std::vector<double>> loadLabelData(std::string path, int numOfImages) {
	std::ifstream file;
	file.open(path, std::ios::binary);

	std::vector<std::vector<double>> labelData;
	labelData.resize(numOfImages);

	std::vector<char> buffer;
	buffer.resize(numOfImages + 8);
	file.read(buffer.data(), numOfImages + 8);

	for(int i = 0; i < numOfImages; i++) {
		for(int j = 0; j < buffer[i + 8]; j++) {
			labelData[i].push_back(0);
		}
		labelData[i].push_back(1);
		for(int j = 0; j < 10 - (buffer[i + 8] + 1); j++) {
			labelData[i].push_back(0);
		}
	}

	return labelData;
}

void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate) {
	learningRate = rate;

	W1 = Matrix(inputNeuron, hiddenNeuron);
	W2 = Matrix(hiddenNeuron, hiddenNeuron);
	W3 = Matrix(hiddenNeuron, outputNeuron);
	B1 = Matrix(1, hiddenNeuron);
	B2 = Matrix(1, hiddenNeuron);
	B3 = Matrix(1, outputNeuron);

	W1 = W1.applyFunction(random);
	W2 = W2.applyFunction(random);
	W3 = W3.applyFunction(random);
	B1 = B1.applyFunction(random);
	B2 = B2.applyFunction(random);
	B3 = B3.applyFunction(random);
}

Matrix computeOutput(std::vector<double> input) {
	X  = Matrix({input});
	H  = X.dot(W1).add(B1).applyFunction(sigmoid);
	H2 = H.dot(W2).add(B2).applyFunction(sigmoid);
	Y  = H2.dot(W3).add(B3).applyFunction(sigmoid);
	return Y;
}

void learn(std::vector<double> expectedOutput) {
	Y2 = Matrix({expectedOutput}); 

	dJdB3 = Y.subtract(Y2).multiply(H2.dot(W3).add(B3).applyFunction(sigmoidePrime));
	dJdB2 = dJdB3.dot(W3.transpose()).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
	dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
	dJdW3 = H2.transpose().dot(dJdB3);
	dJdW2 = H.transpose().dot(dJdB2);
	dJdW1 = X.transpose().dot(dJdB1);

	W1 = W1.subtract(dJdW1.multiply(learningRate));
	W2 = W2.subtract(dJdW2.multiply(learningRate));
	W3 = W3.subtract(dJdW3.multiply(learningRate));
	B1 = B1.subtract(dJdB1.multiply(learningRate));
	B2 = B2.subtract(dJdB2.multiply(learningRate));
	B3 = B3.subtract(dJdB3.multiply(learningRate));
}

double stepFunction(double x) {
	if(x > 0.9) {
		return 1.0;
	}
	if(x < 0.1){
		return 0.0;
	}
	return x;
}

int main(int argc, char** args) {
	std::vector<std::vector<double>> trainingImageData = loadImageData("training_data.idx3-ubyte", 60000);
	std::vector<std::vector<double>> testImageData = loadImageData("test_data.idx3-ubyte", 10000);
	std::vector<std::vector<double>> trainingLabelData = loadLabelData("training_labels.idx1-ubyte", 60000);
	std::vector<std::vector<double>> testLabelData = loadLabelData("test_labels.idx1-ubyte", 10000);
	
	/*for(int i = 0; i < 27; i++) {
		for(int j = 0; j < 27; j++){
			if(trainingImageData[0][i*28+j] != 0) {
				std::cout << "1";
			}else{
				std::cout << "0";
			}
		}
		std::cout << '\n';
	}*/

	srand(time(NULL));

	init(784, 12, 10, 0.001);
	
	int batchSize = 2000;
	int iterationCount = 1;

	for (int i = 0; i < iterationCount; i++) {	
		//Train
		for (int j = 0; j < batchSize; j++) {
			computeOutput(trainingImageData[j]);
			learn(trainingLabelData[j]);
		}
		
		//Test
		int total = 0;
		for (int j = 0; j < 300; j++) {
			Matrix test = computeOutput(testImageData[j]);
			double error;
			for (int k = 0; k < 10; k++) {
				error += test.data[0][k] - testLabelData[j][k];
			}
			if (error < 0.3) {total++;}
		}
		
		std::cout << "#" << i+1 << "/" << iterationCount << " iteration, accuracy: " << (total/300.0) << '\n';
	}

	/*for (int i = 0; i < 300; i++) {
		Matrix test = computeOutput(testImageData[i]);
		double normalizer = *std::max_element(test.data[0].begin(), test.data[0].end());
		for (int j = 0; j < 10; j++) {
			std::cout << testLabelData[i][j] << " ";	 
		}
		std::cout << ": ";
		for (int j = 0; j < 10; j++) {
			std::cout << test.data[0][j] << " ";
			if(normalizer != 0) {
				std::cout << (test.data[0][j] / normalizer) << " ";	 
			}else{
				std::cout << test.data[0][j] << " ";
			}	
		}
		std::cout << '\n';
	}*/

	return 0;
}