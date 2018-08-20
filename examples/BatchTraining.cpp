#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>

#include "NeuralNetwork.h"
#include "Matrix.h"
#include "ExtraFuncs.h"

std::vector<std::vector<double>> readCSV(std::string filepath) {
	std::ifstream file(filepath, std::ifstream::in);
	std::vector<std::vector<double>> data;

	while (file.good()) {
		std::string line;
		std::getline(file, line);

		std::stringstream values(line);

		std::vector<double> valsInLine;
		while (values.good()) {
			std::string newValue;
			std::getline(values, newValue, ',');
			valsInLine.push_back(std::stod(newValue));
		}
		data.push_back(valsInLine);
	}
	file.close();
	return data;
}

std::vector<double> encodeLabel(int val, int possiblities) {
	std::vector<double> label;
	for (int i = 0; i <possiblities; i++) {
		label.push_back(0.0);
	}
	label[val] = 1.0;
	return label;
}

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

template <typename T>
void split(std::vector<T> a, std::vector<T>& outB, std::vector<T>& outC) {
	for (int i = 0; i < ceil(a.size() / 2.f); i++)
		outB.push_back(a[i]);
	for (int i = ceil(a.size() / 2.f); i < a.size(); i++)
		outC.push_back(a[i]);
}

int main() {
	srand(static_cast<unsigned int>(time(NULL)));

	/*// Training data setup
	std::vector<std::vector<double>> irisData = readCSV("iris-data.csv");
	std::vector<std::vector<double>> irisDataLabels = readCSV("iris-labels.csv");

	std::vector<std::vector<double>> trainingData;
	std::vector<std::vector<double>> trainingDataLabels;

	std::vector<std::vector<double>> testingData;
	std::vector<std::vector<double>> testingDataLabels;

	split(irisData, trainingData, testingData);
	split(irisDataLabels, trainingDataLabels, testingDataLabels);

	encodeLabels(trainingDataLabels, getOutputCount(trainingDataLabels));
	encodeLabels(testingDataLabels, getOutputCount(testingDataLabels));

	std::cout << "Training data size: " << trainingData.size() << std::endl;
	std::cout << "Training labels size: " << trainingDataLabels.size() << std::endl;
	std::cout << "Training inputs: " << trainingData[0].size() << std::endl;
	std::cout << "Training outputs: " << trainingDataLabels[0].size() << std::endl;
	std::cout << "Testing data size: " << testingData.size() << std::endl;
	std::cout << "Testing labels size: " << testingDataLabels.size() << std::endl;
	std::cout << "Testing inputs: " << testingData[0].size() << std::endl;
	std::cout << "Testing outputs: " << testingDataLabels[0].size() << std::endl;*/

	int SAMPLES = 1050;
	int BATCH_SIZE = 100;
	int ITERATIONS = 10;

	for (int i = 0; i < ITERATIONS; i++) {
		for (int j = 0; j < SAMPLES / BATCH_SIZE; j++) {
			std::cout << "----" << j * BATCH_SIZE << "->" << (1 + j) * BATCH_SIZE - 1 << "----" << std::endl;
			for (int k = j * BATCH_SIZE; k < (1 + j) * BATCH_SIZE; k++) {
				std::cout << k << std::endl;
			}
		}
		if (SAMPLES % BATCH_SIZE != 0) { // remainder batch
			std::cout << "----" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) << "->" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE) - 1 << "----" << std::endl;
			for (int j = ((SAMPLES / BATCH_SIZE)*BATCH_SIZE); j < ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE); j++) {
				std::cout << j << std::endl;
			}
		}
	}
	
/*
	// Neural Net Parameters
	int hiddenLayers = 8;
	int trainingIterations = 100000;
	double trainingRate = 0.1;
	doubleFunction activationFunc = NeuralNetwork::sigmoid;
	doubleFunction derivativeFunc = NeuralNetwork::sigmoid_d;

	// constructor sets up input and output size based on the data
	//NeuralNetwork nn(trainingData[0].size(), hiddenLayers, trainingDataLabels[0].size(), trainingRate, activationFunc, derivativeFunc);
	NeuralNetwork nn(4, hiddenLayers, 3, trainingRate, activationFunc, derivativeFunc);

	// training loop
	for (int i = 0; i < trainingIterations; i++) {
		int randomIndex = rand() % trainingData.size();
		nn.train(trainingData[randomIndex], trainingDataLabels[randomIndex]);
	}

	// simple validation
	int correct = 0;
	for (int i = 0; i < testingData.size(); i++) {
		std::vector<double> output = nn.predict(testingData[i]).toVector(); // round each element in the matrix to either 0 or 1
		std::vector<double> expectedOutput = testingDataLabels[i];
		
		double ahighest = -1;
		double ahighestIndex = -1;
		for (int i = 0; i < output.size(); i++) {
			if (output[i] > ahighest) {
				ahighest = output[i];
				ahighestIndex = i;
			}
		}

		double bhighest = -1;
		double bhighestIndex = -1;
		for (int i = 0; i < expectedOutput.size(); i++) {
			if (expectedOutput[i] > bhighest) {
				bhighest = expectedOutput[i];
				bhighestIndex = i;
			}
		}

		if (ahighestIndex == bhighestIndex) {
			correct++;
		}
		//std::cout << (ahighestIndex == bhighestIndex ? "Correct!" : "Incorrect.") << std::endl;

		//std::cout << (output == expectedOutput ? "Correct!" : "Incorrect.") << std::endl;
		
		// prints the test case to the console
		//output.print();
	}

	std::cout << ((float)correct / (float)testingData.size()) * 100 << "%" << std::endl;
*/
	// pause
	std::string in;
	std::cin >> in;

	return 0;
}