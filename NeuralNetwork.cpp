/*
Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#include "NeuralNetwork.h"
#include "Matrix.h"

namespace nn {

	/*
		TODO:
		Documentation
	*/
	NeuralNetwork::NeuralNetwork(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double trainingRate, doubleFunction activation, doubleFunction derivative, matrixFunction classifier, matrixFunction classifierDerivative) {
		this->trainingRate = trainingRate;
		this->activation = activation;
		this->derivative = derivative;
		this->classifier = classifier;
		this->classifierDerivative = classifierDerivative;

		weights_ih = Matrix(hiddenLayerNeurons, inputNeurons, true);
		bias_h0 = Matrix(hiddenLayerNeurons, 1, true);

		weights_ho = Matrix(outputNeurons, hiddenLayerNeurons, true);
		bias_o = Matrix(outputNeurons, 1, true);
	}

	/*
		TODO:
		Documentation
	*/
	NeuralNetwork::NeuralNetwork(int inputNeurons, std::vector<int> hiddenLayersShape, int outputNeurons, double trainingRate, doubleFunction activation, doubleFunction derivative, matrixFunction classifier, matrixFunction classifierDerivative) {
		if (hiddenLayersShape.size() == 1) {
			NeuralNetwork(inputNeurons, hiddenLayersShape[0], outputNeurons, trainingRate, activation, derivative, classifier);
			return;
		}

		this->trainingRate = trainingRate;
		this->activation = activation;
		this->derivative = derivative;
		this->classifier = classifier;
		this->classifierDerivative = classifierDerivative;

		weights_ih = Matrix(hiddenLayersShape[0], inputNeurons, true);
		bias_h0 = Matrix(hiddenLayersShape[0], 1, true);

		//for (int i = 1; i < hiddenLayersShape.size(); i++) {
		//	weightsHidden.push_back(Matrix(hiddenLayersShape[i], hiddenLayersShape[i-1], true));
		//	biasesHidden.push_back(Matrix(hiddenLayersShape[i], 1, true));
		//}

		weights_ho = Matrix(outputNeurons, hiddenLayersShape[hiddenLayersShape.size() - 1], true);
		bias_o = Matrix(outputNeurons, 1, true);
	}

	/*
		TODO:
		Documentation
		Feed forward implementation
	*/
	Matrix NeuralNetwork::predict(Matrix inputs) {
		// Feed Forward
		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden0 = Matrix::dot(weights_ih, inputs);
		hidden0 += bias_h0;
		hidden0.map(activation);

		// calculate middle hidden layers
		// hidden = activation((weights*previousLayer)+bias)
		/*std::vector<Matrix> hiddens;
		for(int i = 0; i < weightsHidden.size(); i++) {
		Matrix newHidden = Matrix::dot(weightsHidden[i], (i > 0) ? hiddens[i - 1] : hidden0);
		newHidden += biasesHidden[i];
		newHidden.map(activation);
		hiddens.push_back(newHidden);
		}*/

		// calculate neuron values between the last hidden layer and output
		// output = activation((weights*lastHidden)+bias)
		//Matrix outputs = Matrix::dot(weights_ho, weightsHidden.size() > 1 ? hiddens[hiddens.size()-1] : hidden0);
		Matrix outputs = Matrix::dot(weights_ho, hidden0);
		outputs += bias_o;
		outputs.map(classifier);

		// Return the results
		return outputs;
	}

	/*
		TODO:
		Documentation
		helper function that converts input to matrix before guessing
	*/
	Matrix NeuralNetwork::predict(std::vector<double> inputs) {
		return NeuralNetwork::predict(Matrix::fromVector(inputs));
	}

	/*
		TODO:
		Documentation
	*/
	void NeuralNetwork::train(Matrix inputs, Matrix targets) {
		// Feed Forward
		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden = Matrix::dot(weights_ih, inputs);
		hidden += bias_h0;
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
		bias_h0 += hiddenGradients;
	}

	/*
		TODO:
		Documentation
		helper function that converts parameters to matrices before training
	*/
	void NeuralNetwork::train(std::vector<double> inputs, std::vector<double> targets) {
		NeuralNetwork::train(Matrix::fromVector(inputs), Matrix::fromVector(targets));
	}

	/*
		TODO:
		Documentation
	*/
	void NeuralNetwork::batchTrain(std::vector<Matrix> inputs, std::vector<Matrix> targets, int ITERATIONS, int BATCH_SIZE) {
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
					hidden += bias_h0;
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
				bias_h0 += (squaredErrorSum / BATCH_SIZE) * trainingRate;

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
					hidden += bias_h0;
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
				bias_h0 += (squaredErrorSum / (SAMPLES % BATCH_SIZE)) * trainingRate;

				// prints iteration followed by batch number followed by the mean squared error for this batch
				std::cout << i + 1 << ".rem" << ": " << squaredErrorSum / (SAMPLES % BATCH_SIZE) << std::endl; // mean squared error
			}
		}
	}

	/*
		TODO:
		Documentation
		helper function that converts parameters to matrices before training
	*/
	void NeuralNetwork::batchTrain(std::vector<std::vector<double>> inputsVector, std::vector<std::vector<double>> targetsVector, int ITERATIONS, int BATCH_SIZE) {
		std::vector<Matrix> inputs;
		for (std::vector<double> in : inputsVector)
			inputs.push_back(Matrix::fromVector(in));

		std::vector<Matrix> targets;
		for (std::vector<double> tar : targetsVector)
			targets.push_back(Matrix::fromVector(tar));

		NeuralNetwork::batchTrain(inputs, targets, ITERATIONS, BATCH_SIZE);
	}

	/*
		TODO:
		Documentation
		maps each weight and bias in place
	*/
	void NeuralNetwork::mutate(doubleFunction func, double chance) {
		weights_ih.map(func, chance);
		weights_ho.map(func, chance);
		bias_h0.map(func, chance);
		bias_o.map(func, chance);

		std::cout << "Still using placeholder code in mutate?" << std::endl;
		//weights_h0h1.map(func, chance);
		//bias_h1.map(func, chance);
	}

	/*
		TODO:
		Documentation
		maps each weight and bias returning a copy
	*/
	NeuralNetwork NeuralNetwork::mutate(NeuralNetwork a, doubleFunction func, double chance) {
		NeuralNetwork temp = a;
		temp.mutate(func, chance);
		return temp;
	}
}