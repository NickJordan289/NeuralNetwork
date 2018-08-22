#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>

#include "NeuralNetwork.h"
#include "Matrix.h"
#include "ExtraFuncs.h"

namespace DataSet {
	/*
		Reads the given file seperating into lines and values
		seperates lines using the '\n' delimiter 
		and seperates values using the given delim char (default ',')
	*/
	std::vector<std::vector<double>> read(std::string filepath, char delim=',') {
		std::ifstream file(filepath, std::ifstream::in);
		std::vector<std::vector<double>> data;

		while (file.good()) {
			std::string line;
			std::getline(file, line);

			std::stringstream values(line);

			std::vector<double> valsInLine;
			while (values.good()) {
				std::string newValue;
				std::getline(values, newValue, delim);
				valsInLine.push_back(std::stod(newValue));
			}
			data.push_back(valsInLine);
		}
		file.close();
		return data;
	}

	/*
		Converts integer to a vector filled with zeros
		eg. val=1, possibilities=3
		[0, 1, 0], index val is 1 where all other entries are 0
	*/
	std::vector<double> encodeLabel(int val, int possiblities) {
		std::vector<double> label;
		for (int i = 0; i < possiblities; i++) {
			label.push_back(0.0);
		}
		label[val] = 1.0;
		return label;
	}

	/*
		Encodes a dataset of integers into their respective output possibility chance
		see above function for what a better explanation of what is actually happening

		example. vals = {0, 1, 2, 0}
		returns {{1,0,0},{0,1,0},{0,0,1},{1,0,0}}
	*/
	std::vector<std::vector<double>> encodeLabels(std::vector<std::vector<double>>& vals, int possiblities) {
		std::vector<std::vector<double>> labels;
		for (std::vector<double> a : vals) {
			for (double val : a) {
				labels.push_back(encodeLabel(val, possiblities));
			}
		}
		vals = labels;
		return labels;
	}

	/*
		Hacky solution to determining the amount of unique outputs in a given label vector
		Sorts so the values go from min to max then counts only numbers it hasn't seen
	*/
	int getOutputCount(std::vector<std::vector<double>> labels) {
		std::sort(labels.begin(), labels.end());
		int max = -1;
		int count = 0;
		for (std::vector<double> line : labels) {
			for (double value : line) {
				if (value > max) {
					max = value;
					count++;
				}
			}
		}
		return count;
	}

	/*
		Splits a given vector into two vectors
		Split ratio is given by ratioA and ratioB
		default split ratio of 1/2
	*/
	template <typename T>
	void split(std::vector<T> a, std::vector<T>& outB, std::vector<T>& outC, int ratioA = 1, int ratioB = 1) {
		int totalRatio = ratioA + ratioB;
		float percent = (float)ratioA / (float)totalRatio;
		for (int i = 0; i < ceil(a.size() * percent); i++)
			outB.push_back(a[i]);
		for (int i = ceil(a.size() * percent); i < a.size(); i++)
			outC.push_back(a[i]);
	}
}

int main() {
	// Training data setup
	std::vector<std::vector<double>> irisData = DataSet::read("iris-data.csv", ',');
	std::vector<std::vector<double>> irisDataLabels = DataSet::read("iris-labels.csv", ',');

	// initial fixed seed shuffle
	// doing this because the input data set is sorted
	srand(1); 
	std::random_shuffle(std::begin(irisData), std::end(irisData));
	srand(1);
	std::random_shuffle(std::begin(irisDataLabels), std::end(irisDataLabels));

	std::vector<std::vector<double>> trainingData;
	std::vector<std::vector<double>> trainingDataLabels;

	std::vector<std::vector<double>> testingData;
	std::vector<std::vector<double>> testingDataLabels;

	// split data 2/3 to training, 1/3 to testing
	DataSet::split(irisData, trainingData, testingData, 2, 1);
	DataSet::split(irisDataLabels, trainingDataLabels, testingDataLabels, 2, 1);

	// convert from 0, 1, 2 to [1,0,0], [0,1,0], [0,0,1]
	DataSet::encodeLabels(trainingDataLabels, DataSet::getOutputCount(trainingDataLabels));
	DataSet::encodeLabels(testingDataLabels, DataSet::getOutputCount(testingDataLabels));

	std::cout << "Training data size: " << trainingData.size() << std::endl;
	std::cout << "Training labels size: " << trainingDataLabels.size() << std::endl;
	std::cout << "Training inputs: " << trainingData[0].size() << std::endl;
	std::cout << "Training outputs: " << trainingDataLabels[0].size() << std::endl;
	std::cout << "Testing data size: " << testingData.size() << std::endl;
	std::cout << "Testing labels size: " << testingDataLabels.size() << std::endl;
	std::cout << "Testing inputs: " << testingData[0].size() << std::endl;
	std::cout << "Testing outputs: " << testingDataLabels[0].size() << std::endl;
	std::cout << std::endl;

	srand(static_cast<unsigned int>(time(NULL)));

	// Neural Net Parameters
	int hiddenLayers = 3;
	doubleFunction activationFunc = NeuralNetwork::sigmoid;
	doubleFunction derivativeFunc = NeuralNetwork::sigmoid_d;
	matrixFunction classifierFunc = NeuralNetwork::softmax;
	
	// Training Parameters
	double trainingRate = 0.05;
	int BATCH_SIZE = 10;
	int ITERATIONS = 200;
	
	// constructor sets up input and output size based on the data
	NeuralNetwork nn(trainingData[0].size(), hiddenLayers, trainingDataLabels[0].size(), trainingRate, activationFunc, derivativeFunc, classifierFunc);

	// simple batch train that uses mse to calculate biases
	nn.batchTrain(trainingData, trainingDataLabels, ITERATIONS, BATCH_SIZE);

	// simple validation
	int correct = 0;
	for (int i = 0; i < testingData.size(); i++) {
		Matrix output = nn.predict(testingData[i]).map(round); // predicts then rounds output
		Matrix expectedOutput = Matrix::fromVector(testingDataLabels[i]);

		if (output == expectedOutput) {
			correct++;
		}
		std::cout << (output == expectedOutput ? "Correct!" : "Incorrect.") << std::endl;

		// prints the test case to the console
		output.print();
		expectedOutput.print();
	}
	std::cout << correct << " / " << testingDataLabels.size() << " = " << ((float)correct / (float)testingDataLabels.size()) * 100 << "%" << std::endl;

	// pause
	std::string in;
	std::cin >> in;

	return 0;
}
	