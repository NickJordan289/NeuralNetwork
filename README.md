[![Build Status](https://travis-ci.org/NickJordan289/NeuralNetwork.svg?branch=master)](https://travis-ci.org/NickJordan289/NeuralNetwork)

# What is this repo for?
Simple neural network library written in C++ that implements a multilayer perceptron

[Latest Builds](https://github.com/NickJordan289/NeuralNetwork/releases)

# Examples
[MNIST](examples/MNIST)

[Logical Operators](examples/LogicalOperators.cpp)

[Flappy Bird NEAT](https://github.com/NickJordan289/NEAT-Games/tree/master/FlappyBird)

[Iris Flower Dataset](examples/IrisFlowerDataSet.cpp)

Credit to Daniel Shiffman's videos on [Perceptrons and Neural Networks](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb).

# How to Use
1. Download the source, lib and dll
2. Link to your project
3. 
```cpp
#include "NNLib.hpp"
```

### Basic Instantiation
```cpp
ml::NeuralNetwork nn = ml::NeuralNetwork(2, { 4 }, 1, 0.1, ml::SIGMOID, ml::SIGMOID);
```

### Basic Training
```cpp
for (int i = 0; i < 100000; i++) {
	int r = rand() % XORtrainingData.size();
	nn.train(XORtrainingData[r], XORtrainingDataLabels[r]);
}
```

### Basic Testing
```cpp
for (int i = 0; i < XORtrainingDataLabels.size(); i++) {
	auto output = nn.predict(XORtrainingData[i]);
	output.print();
	std::cout << "Error: " << output.sum() - XORtrainingDataLabels[i][0] << std::endl;
	std::cout << (output.map(round) == XORtrainingDataLabels[i] ? "Correct" : "Incorrect") << std::endl;
}
```

### Loading a neural network from a file
```cpp
ml::NeuralNetwork nn;
try {
	nn = ml::NeuralNetwork::LoadFromFile("MNIST.txt");
} catch (const std::exception& e) {
	std::cout << "Error reading saved network, falling back to new network" << std::endl;
	nn = ml::NeuralNetwork(784, { 16, 16 }, 10, 0.1, ml::SIGMOID, ml::SIGMOID);
}
```

### Saving a neural network to a file
```cpp
nn.saveToFile("nn.txt");
```

## DataFuncs
### Loading from text file
```cpp
// Reads iris-data.csv with the ',' delimiter
std::vector<std::vector<double>> irisData = ml::DataFuncs::loadFromFile("iris-data.csv", ',');
```

### Splitting vector into training and testing
```cpp
// this example splits 2/3 to trainingData and 1/3 to testingData
std::vector<std::vector<double>> trainingData, testingData, trainingDataLabels, testingDataLabels;
ml::DataFuncs::splitVector(irisData, trainingData, testingData, 2, 1);
ml::DataFuncs::splitVector(irisDataLabels, trainingDataLabels, testingDataLabels, 2, 1);
```

### Encoding labels
### Before
| index | value |  
| ----- |:-----:|
| 0     | 2     |
| 1     | 1     |
| 2     | 3     |
| 3     | 0     |
```cpp
ml::DataFuncs::encodeLabels(trainingDataLabels, ml::DataFuncs::getOutputCount(trainingDataLabels));
ml::DataFuncs::encodeLabels(testingDataLabels, ml::DataFuncs::getOutputCount(testingDataLabels));
```
### After
| index |   value   |  
| ----- |:---------:|
| 0     | {0,0,1,0} |
| 1     | {0,1,0,0} |
| 2     | {0,0,0,1} |
| 3     | {1,0,0,0} |
