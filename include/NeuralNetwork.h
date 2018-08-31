/*
	Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#pragma once

#ifdef TESTLIBRARY_EXPORTS  
#define TESTLIBRARY_API __declspec(dllexport)   
#else  
#define TESTLIBRARY_API __declspec(dllimport)   
#endif  

#include "Matrix.h"

namespace ml {

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

	inline static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	inline static double sigmoid_d(double x) {
		return x * (1.0 - x);
	}

	inline static double sigmoid(double x, Matrix) {
		return 1.0 / (1.0 + exp(-x));
	}

	inline static double sigmoid_d(double x, Matrix) {
		return x * (1.0 - x);
	}

	inline static double softmax(double x, Matrix a) {
		return exp(x) / a.map(exp).sum();
	}

	inline static double softmax_d(double x, Matrix a) {
		throw std::exception("Not implemented");
		//return a.map(exp).sum();
	}

	inline static double tanh(double x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	inline static double tanh_d(double x) {
		return 1 - pow(tanh(x), 2);
	}

	inline static double tanh(double x, Matrix) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}

	inline static double tanh_d(double x, Matrix) {
		return 1 - pow(tanh(x), 2);
	}

	inline static double relu(double x) {
		return (x < 0) ? 0 : x;
	}

	inline static double relu_d(double x, Matrix) {
		return (x < 0) ? 0 : 1;
	}

	inline static double relu(double x, Matrix) {
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

	inline static double l_relu(double x, Matrix) {
		return (x < 0) ? 0.01*x : x;
	}

	inline static double l_relu_d(double x, Matrix) {
		return (x < 0) ? 0.01 : 1;
	}

	class NeuralNetwork {
	private:
		Matrix weights_ih, weights_ho, bias_h0, bias_o;
		double trainingRate;
		doubleFunction activation;
		doubleFunction derivative;
		matrixFunction classifier;
		matrixFunction classifierDerivative;

		//std::vector<Matrix> weightsHidden;
		//std::vector<Matrix> biasesHidden;

	public:
		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API NeuralNetwork(int, int, int, double=0.1, doubleFunction=ml::sigmoid, doubleFunction=ml::sigmoid_d, matrixFunction=ml::sigmoid, matrixFunction=ml::sigmoid_d);
	
		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API NeuralNetwork(int, std::vector<int>, int, double=0.1, doubleFunction=ml::sigmoid, doubleFunction=ml::sigmoid_d, matrixFunction=ml::sigmoid, matrixFunction=ml::sigmoid_d);

		/*
			TODO:
			Documentation
			Feed forward implementation
		*/
		TESTLIBRARY_API Matrix predict(Matrix inputs);
	
		/*
			TODO:
			Documentation
			helper function that converts input to matrix before guessing
		*/
		TESTLIBRARY_API Matrix predict(std::vector<double> inputs);

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API void train(Matrix inputs, Matrix targets);

		/*
			TODO:
			Documentation
			helper function that converts parameters to matrices before training
		*/
		TESTLIBRARY_API void train(std::vector<double> inputs, std::vector<double> targets);

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API void batchTrain(std::vector<Matrix> inputs, std::vector<Matrix> targets, int ITERATIONS, int BATCH_SIZE);

		/*
			TODO:
			Documentation
			helper function that converts parameters to matrices before training
		*/
		TESTLIBRARY_API void batchTrain(std::vector<std::vector<double>> inputsVector, std::vector<std::vector<double>> targetsVector, int ITERATIONS, int BATCH_SIZE);

		/*
			TODO:
			Documentation
			maps each weight and bias in place
		*/
		TESTLIBRARY_API void mutate(doubleFunction func, double chance);

		/*
			TODO:
			Documentation
			maps each weight and bias returning a copy
		*/
		TESTLIBRARY_API static NeuralNetwork mutate(NeuralNetwork a, doubleFunction func, double chance);
	};
	}