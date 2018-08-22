/*
	Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#pragma once
#include "Matrix.h"
#include <vector>
#include "ExtraFuncs.h"

// function pointer type
// type name : doubleFunction
// param1 : double
// return : double
typedef double(*doubleFunction)(double);

// function pointer type
// type name : matrixFunction
// param1 : double
// param2 : Matrix
// return : double
typedef double(*matrixFunction)(double,Matrix);

class NeuralNetwork {
private:
	Matrix weights_ih, weights_ho, bias_h, bias_o;
	double trainingRate;
	doubleFunction activation;
	doubleFunction derivative;
	matrixFunction classifier;

public:
	inline static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	inline static double sigmoid_d(double x) {
		return x * (1.0 - x);
	}

	inline static double sigmoid(double x, Matrix) {
		return 1.0 / (1.0 + exp(-x));
	}

	inline static double softmax(double x, Matrix a) {
		return exp(x) / a.map(exp).sum();
	}

	inline static double tanh(double x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	inline static double tanh_d(double x) {
		return 1 - pow(tanh(x),2);
	}

	inline static double relu (double x) {
		return (x < 0) ? 0 : x;
	}

	inline static double relu_d(double x) {
		return (x < 0) ? 0 : 1;
	}

	inline static double l_relu(double x) {
		return (x < 0) ? 0.01*x : x;
	}

	inline static double l_relu_d(double x) {
		return (x < 0) ? 0.01 : 1;
	}

	NeuralNetwork(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double trainingRate=0.1, doubleFunction activation=sigmoid, doubleFunction derivative=sigmoid_d, matrixFunction classifier=sigmoid) {
		this->trainingRate = trainingRate;
		this->activation = activation;
		this->derivative = derivative;
		this->classifier = classifier;

		weights_ih = Matrix(hiddenLayerNeurons, inputNeurons, true);
		weights_ho = Matrix(outputNeurons, hiddenLayerNeurons, true);
		bias_h = Matrix(hiddenLayerNeurons, 1, true);
		bias_o = Matrix(outputNeurons, 1, true);
	}

	// Feed forward implementation
	Matrix predict(Matrix inputs) {
	// Feed Forward
		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden = Matrix::dot(weights_ih, inputs);
		hidden += bias_h;
		hidden.map(activation);

		// calculate neuron values between hidden and output
		// output = activation((weights*hidden)+bias)
		Matrix outputs = Matrix::dot(weights_ho, hidden);
		outputs += bias_o;
		outputs.map(classifier);

		// Return the result as an array rather than a single column matrix
		return outputs;
	}
	
	void train(Matrix inputs, Matrix targets, bool debug=false) {
	// Feed Forward
		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden = Matrix::dot(weights_ih, inputs);
		hidden += bias_h;
		hidden.map(activation);

		// calculate neuron values between hidden and output
		// output = activation((weights*hidden)+bias)
		Matrix outputs = Matrix::dot(weights_ho, hidden);
		outputs += bias_o;
		outputs.map(classifier);

	// Back Propagation
	// Hidden to Output Weights

		// error = how far off the output was
		// gradient = how much each weight influenced the output
		Matrix outputErrors = targets - outputs;
		Matrix outputGradients = Matrix::Map(outputs, derivative);
		outputGradients *= outputErrors;
		outputGradients *= trainingRate;

		// hidden->output deltas
		Matrix hoAdjustments = Matrix::dot(outputGradients, hidden.T());
		weights_ho += hoAdjustments;
		bias_o += outputGradients;

	// Input to Hidden Weights

		// error = how far off the output was
		// gradient = how much each weight influenced the output
		Matrix hiddenErrors = Matrix::dot(weights_ho.T(), outputErrors);
		Matrix hiddenGradients = Matrix::Map(hidden, derivative);
		hiddenGradients *= hiddenErrors;
		hiddenGradients *= trainingRate;

		// input->hidden deltas
		Matrix ihAdjustments = Matrix::dot(hiddenGradients, inputs.T());
		weights_ih += ihAdjustments;
		bias_h += hiddenGradients;
	}

	void batchTrain(std::vector<Matrix> inputs, std::vector<Matrix> targets, int ITERATIONS, int BATCH_SIZE) {
		int SAMPLES = inputs.size();
		for (int i = 0; i < ITERATIONS; i++) {
			// shuffle data
			srand(ITERATIONS);
			std::random_shuffle(std::begin(inputs), std::end(inputs));
			srand(ITERATIONS);
			std::random_shuffle(std::begin(targets), std::end(targets));

			for (int j = 0; j < SAMPLES / BATCH_SIZE; j++) {
				//std::cout << "--------" << j * BATCH_SIZE << "->" << (1 + j) * BATCH_SIZE - 1 << "--------" << std::endl;
				double squaredErrorSum = 0.0;
				for (int k = j * BATCH_SIZE; k < (1 + j) * BATCH_SIZE; k++) {

				// Feed Forward
					// calculate neuron values between input and hidden
					// hidden = activation((weights*input)+bias)
					Matrix hidden = Matrix::dot(weights_ih, inputs[k]);
					hidden += bias_h;
					hidden.map(activation);

					// calculate neuron values between hidden and output
					// output = activation((weights*hidden)+bias)
					Matrix outputs = Matrix::dot(weights_ho, hidden);
					outputs.map(classifier);

				// Back Propagation
				// Hidden to Output Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix outputErrors = targets[k] - outputs;
					for (double err : outputErrors.toVector()) {
						squaredErrorSum += err * err;
					}

					Matrix outputGradients = Matrix::Map(outputs, derivative);
					outputGradients *= outputErrors;
					outputGradients *= trainingRate;

					// hidden->output deltas
					Matrix hoAdjustments = Matrix::dot(outputGradients, hidden.T());
					weights_ho += hoAdjustments;

				// Input to Hidden Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix hiddenErrors = Matrix::dot(weights_ho.T(), outputErrors);
					Matrix hiddenGradients = Matrix::Map(hidden, derivative);
					hiddenGradients *= hiddenErrors;
					hiddenGradients *= trainingRate;

					// input->hidden deltas
					Matrix ihAdjustments = Matrix::dot(hiddenGradients, inputs[k].T());
					weights_ih += ihAdjustments;
				}
				bias_o += (squaredErrorSum / BATCH_SIZE) * trainingRate;
				bias_h += (squaredErrorSum / BATCH_SIZE) * trainingRate;

				// prints iteration followed by batch number followed by the mean squared error for this batch
				std::cout << i + 1 << "." << j << ": " << squaredErrorSum / BATCH_SIZE << std::endl; // mean squared error
			}
			if (SAMPLES % BATCH_SIZE != 0) { // remainder batch
				//std::cout << "----" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) << "->" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE) - 1 << "----" << std::endl;
				double squaredErrorSum = 0.0;
				for (int k = ((SAMPLES / BATCH_SIZE)*BATCH_SIZE); k < ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE); k++) {

					// Feed Forward
					// calculate neuron values between input and hidden
					// hidden = activation((weights*input)+bias)
					Matrix hidden = Matrix::dot(weights_ih, inputs[k]);
					hidden += bias_h;
					hidden.map(activation);

					// calculate neuron values between hidden and output
					// output = activation((weights*hidden)+bias)
					Matrix outputs = Matrix::dot(weights_ho, hidden);
					outputs.map(classifier);

					// Back Propagation
					// Hidden to Output Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix outputErrors = targets[k] - outputs;
					for (double err : outputErrors.toVector()) {
						squaredErrorSum += err * err;
					}

					Matrix outputGradients = Matrix::Map(outputs, derivative);
					outputGradients *= outputErrors;
					outputGradients *= trainingRate;

					// hidden->output deltas
					Matrix hoAdjustments = Matrix::dot(outputGradients, hidden.T());
					weights_ho += hoAdjustments;

					// Input to Hidden Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix hiddenErrors = Matrix::dot(weights_ho.T(), outputErrors);
					Matrix hiddenGradients = Matrix::Map(hidden, derivative);
					hiddenGradients *= hiddenErrors;
					hiddenGradients *= trainingRate;

					// input->hidden deltas
					Matrix ihAdjustments = Matrix::dot(hiddenGradients, inputs[k].T());
					weights_ih += ihAdjustments;
				}
				bias_o += (squaredErrorSum / (SAMPLES % BATCH_SIZE)) * trainingRate;
				bias_h += (squaredErrorSum / (SAMPLES % BATCH_SIZE)) * trainingRate;

				// prints iteration followed by batch number followed by the mean squared error for this batch
				std::cout << i + 1 << ".rem" << ": " << squaredErrorSum / (SAMPLES % BATCH_SIZE) << std::endl; // mean squared error
			}
		}
	}

	// helper function that converts parameters to matrices before training
	inline void batchTrain(std::vector<std::vector<double>> inputsVector, std::vector<std::vector<double>> targetsVector, int ITERATIONS, int BATCH_SIZE) {
		std::vector<Matrix> inputs;
		for (std::vector<double> in : inputsVector)
			inputs.push_back(Matrix::fromVector(in));
		
		std::vector<Matrix> targets;
		for (std::vector<double> tar : targetsVector)
			targets.push_back(Matrix::fromVector(tar));

		batchTrain(inputs, targets, ITERATIONS, BATCH_SIZE);
	}

	// helper function that converts input to matrix before guessing
	inline Matrix predict(std::vector<double> inputs) {
		return predict(Matrix::fromVector(inputs));
	}

	// helper function that converts parameters to matrices before training
	inline void train(std::vector<double> inputs, std::vector<double> targets, bool debug = false) {
		train(Matrix::fromVector(inputs), Matrix::fromVector(targets), debug);
	}

	// maps each weight and bias in place
	inline void mutate(doubleFunction func, double chance) {
		weights_ih.map(func, chance);
		weights_ho.map(func, chance);
		bias_h.map(func, chance);
		bias_o.map(func, chance);
	}

	// maps each weight and bias returning a copy
	static inline NeuralNetwork mutate(NeuralNetwork a, doubleFunction func, double chance) {
		NeuralNetwork temp = a;
		temp.mutate(func, chance);
		return temp;
	}
};