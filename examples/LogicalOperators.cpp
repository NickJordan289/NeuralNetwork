#include <iostream>
#include <vector>
#include <time.h>
#include <string>

#include "NeuralNetwork.h"
#include "Matrix.h"
#include "ExtraFuncs.h"

int main() {
	srand(static_cast<unsigned int>(time(NULL)));

#pragma region LogicalTrainingSets
	std::vector<std::vector<double>> NOTtrainingData = { { 0 },{ 1 } };
	std::vector<std::vector<double>> NOTtrainingDataLabels = { { 0 },{ 1 } };

	std::vector<std::vector<double>> ORtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> ORtrainingDataLabels = { { 0 },{ 1 },{ 1 },{ 1 } };

	std::vector<std::vector<double>> ANDtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> ANDtrainingDataLabels = { { 0 },{ 0 },{ 0 },{ 1 } };

	std::vector<std::vector<double>> NANDtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> NANDtrainingDataLabels = { { 1 },{ 1 },{ 1 },{ 0 } };

	std::vector<std::vector<double>> NORtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> NORtrainingDataLabels = { { 1 },{ 0 },{ 0 },{ 0 } };

	std::vector<std::vector<double>> XNORtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> XNORtrainingDataLabels = { { 1 },{ 0 },{ 0 },{ 1 } };

	std::vector<std::vector<double>> XORtrainingData = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	std::vector<std::vector<double>> XORtrainingDataLabels = { { 0 },{ 1 },{ 1 },{ 0 } };
#pragma endregion

	// Training to use
	auto trainingSet = XORtrainingData;
	auto trainingSetLabels = XORtrainingDataLabels;

	// Neural Net Parameters
	int hiddenLayers = 3;
	int trainingIterations = 10000;
	float trainingRate = 0.1f;
	doubleFunction activationFunc = NeuralNetwork::sigmoid;
	doubleFunction derivativeFunc = NeuralNetwork::sigmoid_d;

	// constructor sets up input and output size based on the data
	NeuralNetwork nn(trainingSet[0].size(), hiddenLayers, trainingSetLabels[0].size(), trainingRate, activationFunc, derivativeFunc);
	
	// training loop
	for (int i = 0; i < trainingIterations; i++) {
		int randomIndex = rand() % trainingSet.size();
		nn.train(trainingSet[randomIndex], trainingSetLabels[randomIndex]);
	}

	// simple validation
	for (int i = 0; i < trainingSet.size(); i++) {
		Matrix output = nn.predict(trainingSet[i]).map(round); // round each element in the matrix to either 0 or 1
		std::vector<double> expectedOutput = trainingSetLabels[i];

		std::cout << (output == expectedOutput ? "Correct!" : "Incorrect.") << std::endl;
		
		// prints the test case to the console
		std::cout << trainingSetLabels[i][0] << " == ";
		output.print();
	}

	// pause
	std::string in;
	std::cin >> in;

	return 0;
}
