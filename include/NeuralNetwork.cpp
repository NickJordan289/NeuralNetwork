/*
Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#include "NeuralNetwork.h"
#include "Matrix.h"

namespace ml {
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
		bias_h0 = ml::randomDouble(-1, 1);

		weights_ho = Matrix(outputNeurons, hiddenLayerNeurons, true);
		bias_o = ml::randomDouble(-1, 1);
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
		bias_h0 = randomDouble(-1, 1);

		for (int i = 1; i < hiddenLayersShape.size(); i++) {
			weightsHidden.push_back(Matrix(hiddenLayersShape[i], hiddenLayersShape[i-1], true));
			biasesHidden.push_back(randomDouble(-1, 1));
		}

		weights_ho = Matrix(outputNeurons, hiddenLayersShape[hiddenLayersShape.size() - 1], true);
		bias_o = randomDouble(-1, 1);

		this->hiddenLayersShape = hiddenLayersShape; // leaving this in for code clarity despite redudancy
		this->shape = std::make_tuple(inputNeurons, hiddenLayersShape, outputNeurons);
	}


	/*
		TODO:
		Documentation
		Feed Forward Prediction
	*/
	Matrix NeuralNetwork::predict(Matrix inputs) {
		// Feed Forward

		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden0 = Matrix::Dot(weights_ih, inputs);
		hidden0 += bias_h0;
		hidden0.map(activation);

		// calculate hidden layers after hidden0
		// hidden = activation((weights*previousLayer)+bias)
		std::vector<Matrix> hiddens;	
		if (hiddenLayersShape.size() > 1) {
			for (int i = 0; i < hiddenLayersShape.size() - 1; i++) {
				Matrix hidden = Matrix::Dot(weightsHidden[i], (i > 0) ? hiddens[hiddens.size() - 1] : hidden0); // (weightsToLayer, prevLayer)
				hidden += biasesHidden[i];
				hidden.map(activation);
				hiddens.push_back(hidden);
			}
		}

		// calculate neuron values between the last hidden layer and output
		// output = activation((weights*lastHidden)+bias)
		Matrix outputs = Matrix::Dot(weights_ho, (hiddens.size()>0) ? hiddens[hiddens.size() - 1] : hidden0);
		outputs += bias_o;
		outputs.map(classifier);

		// return result as Matrix
		return outputs;
	}

	/*
		TODO:
		Documentation
		helper function that converts input to matrix before guessing
	*/
	Matrix NeuralNetwork::predict(std::vector<double> inputs) {
		return NeuralNetwork::predict(Matrix::FromVector(inputs));
	}

	/*
		TODO:
		Documentation
	*/
	void NeuralNetwork::train(Matrix inputs, Matrix targets) {
		// Feed Forward

		// calculate neuron values between input and hidden
		// hidden = activation((weights*input)+bias)
		Matrix hidden0 = Matrix::Dot(weights_ih, inputs);
		hidden0 += bias_h0;
		hidden0.map(activation);

		// calculate hidden layers after hidden0
		// hidden = activation((weights*previousLayer)+bias)
		std::vector<Matrix> hiddens;
		if (hiddenLayersShape.size() > 1) {
			for (int i = 0; i < hiddenLayersShape.size() - 1; i++) {
				Matrix hidden = Matrix::Dot(weightsHidden[i], (i > 0) ? hiddens[hiddens.size() - 1] : hidden0); // (weightsToLayer, prevLayer)
				hidden += biasesHidden[i];
				hidden.map(activation);
				hiddens.push_back(hidden);
			}
		}

		// calculate neuron values between the last hidden layer and output
		// output = activation((weights*lastHidden)+bias)
		Matrix outputs = Matrix::Dot(weights_ho, (hiddens.size()>0) ? hiddens[hiddens.size() - 1] : hidden0);
		outputs += bias_o;
		outputs.map(classifier);

		// Back prop
	
			// h1->o weights
			// this is always the same (input to first hidden)

			// error = how far off the output was
			// gradient = how much each weight influenced the output
			Matrix outputErrors = targets - outputs;
			Matrix outputGradients = Matrix::Map(outputs, classifierDerivative);
			outputGradients *= outputErrors;
			outputGradients *= trainingRate;

			// hidden->output deltas
			Matrix hoDeltas = Matrix::Dot(outputGradients, ((hiddens.size()>0) ? hiddens[hiddens.size() - 1] : hidden0).T());
			weights_ho += hoDeltas;
			bias_o += outputGradients.sum() / outputGradients.toVector().size(); // bias = average of the gradients

			// store the last error because we will modifying it inside the loop and outside
			Matrix lastError = outputErrors;

			// sub 1 because we want index
			// i > 0 because 0 is hidden0 which is computed seperately
			for (int i = hiddenLayersShape.size() - 1; i > 0; i--) { 
				// VARIABLES TO MAKE THINGS MORE CLEAR
				Matrix THISLAYER = hiddens[i - 1]; // returns last entry for the first run
				Matrix NEXTLAYER = (i - 1) > 0 ? hiddens[i - 2] : hidden0; // returns -1 on last run therefore falling back to hidden0
				Matrix PREVWEIGHTS = (i == hiddenLayersShape.size() - 1) ? weights_ho : weightsHidden[i]; // weights to the right
				// VARIABLES TO MAKE THINGS MORE CLEAR

				Matrix gradients = Matrix::Map(THISLAYER, derivative);
				Matrix errors = Matrix::Dot(PREVWEIGHTS.T(), lastError);
				gradients *= errors;
				gradients *= trainingRate;
				Matrix deltas = Matrix::Dot(gradients, NEXTLAYER.T());

				// divide by its layer count, further in layers have less control over the output (may be wrong)
				weightsHidden[i-1] += deltas/(i+1);
				biasesHidden[i-1] += gradients.sum() / gradients.toVector().size(); // bias = average of the gradients

				// update error
				lastError = errors;
			}

			// i->h0 weights
			// this is always the same (input to first hidden)

			// error = how far off the output was
			// gradient = how much each weight influenced the output
			Matrix hiddenErrors = Matrix::Dot((weightsHidden.size() > 0 ? weightsHidden[0] : weights_ho).T(), lastError);
			Matrix hiddenGradients = Matrix::Map(hidden0, derivative);
			hiddenGradients *= hiddenErrors;
			hiddenGradients *= trainingRate;

			// input->hidden deltas
			Matrix ihDeltas = Matrix::Dot(hiddenGradients, inputs.T());
			weights_ih += ihDeltas;
			bias_h0 += hiddenGradients.sum() / hiddenGradients.toVector().size(); // bias = average of the gradients
	}

	/*
		TODO:
		Documentation
		helper function that converts parameters to matrices before training
	*/
	void NeuralNetwork::train(std::vector<double> inputs, std::vector<double> targets) {
		NeuralNetwork::train(Matrix::FromVector(inputs), Matrix::FromVector(targets));
	}

	/*
		TODO:
		Documentation
	*/
	void NeuralNetwork::batchTrain(std::vector<Matrix> inputs, std::vector<Matrix> targets, int ITERATIONS, int BATCH_SIZE) {
		int SAMPLES = inputs.size();
		for (int i = 0; i < ITERATIONS; i++) {
			// shuffle data
			srand(i);
			std::random_shuffle(std::begin(inputs), std::end(inputs));
			srand(i);
			std::random_shuffle(std::begin(targets), std::end(targets));

			for (int j = 0; j < SAMPLES / BATCH_SIZE; j++) {
				//std::cout << "--------" << j * BATCH_SIZE << "->" << (1 + j) * BATCH_SIZE - 1 << "--------" << std::endl;
				double squaredErrorSum = 0.0;
				for (int k = j * BATCH_SIZE; k < (1 + j) * BATCH_SIZE; k++) {

					// Feed Forward
					// calculate neuron values between input and hidden
					// hidden = activation((weights*input)+bias)
					Matrix hidden = Matrix::Dot(weights_ih, inputs[k]);
					hidden += bias_h0;
					hidden.map(activation);

					// calculate neuron values between hidden and output
					// output = activation((weights*hidden)+bias)
					Matrix outputs = Matrix::Dot(weights_ho, hidden);
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
					Matrix hoAdjustments = Matrix::Dot(outputGradients, hidden.T());
					weights_ho += hoAdjustments;

					// Input to Hidden Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix hiddenErrors = Matrix::Dot(weights_ho.T(), outputErrors);
					Matrix hiddenGradients = Matrix::Map(hidden, derivative);
					hiddenGradients *= hiddenErrors;
					hiddenGradients *= trainingRate;

					// input->hidden deltas
					Matrix ihAdjustments = Matrix::Dot(hiddenGradients, inputs[k].T());
					weights_ih += ihAdjustments;
				}
				bias_o += (squaredErrorSum / BATCH_SIZE) * trainingRate;
				bias_h0 += (squaredErrorSum / BATCH_SIZE) * trainingRate;

				#ifdef DEBUG
					// prints iteration followed by batch number followed by the mean squared error for this batch
					std::cout << i + 1 << "." << j << ": " << squaredErrorSum / BATCH_SIZE << std::endl; // mean squared error
				#endif
			}
			if (SAMPLES % BATCH_SIZE != 0) { // remainder batch
			//std::cout << "----" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) << "->" << ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE) - 1 << "----" << std::endl;
				double squaredErrorSum = 0.0;
				for (int k = ((SAMPLES / BATCH_SIZE)*BATCH_SIZE); k < ((SAMPLES / BATCH_SIZE)*BATCH_SIZE) + (SAMPLES % BATCH_SIZE); k++) {

					// Feed Forward
					// calculate neuron values between input and hidden
					// hidden = activation((weights*input)+bias)
					Matrix hidden = Matrix::Dot(weights_ih, inputs[k]);
					hidden += bias_h0;
					hidden.map(activation);

					// calculate neuron values between hidden and output
					// output = activation((weights*hidden)+bias)
					Matrix outputs = Matrix::Dot(weights_ho, hidden);
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
					Matrix hoAdjustments = Matrix::Dot(outputGradients, hidden.T());
					weights_ho += hoAdjustments;

					// Input to Hidden Weights

					// error = how far off the output was
					// gradient = how much each weight influenced the output
					Matrix hiddenErrors = Matrix::Dot(weights_ho.T(), outputErrors);
					Matrix hiddenGradients = Matrix::Map(hidden, derivative);
					hiddenGradients *= hiddenErrors;
					hiddenGradients *= trainingRate;

					// input->hidden deltas
					Matrix ihAdjustments = Matrix::Dot(hiddenGradients, inputs[k].T());
					weights_ih += ihAdjustments;
				}
				bias_o += (squaredErrorSum / (SAMPLES % BATCH_SIZE)) * trainingRate;
				bias_h0 += (squaredErrorSum / (SAMPLES % BATCH_SIZE)) * trainingRate;

				#ifdef DEBUG
					// prints iteration followed by batch number followed by the mean squared error for this batch
					std::cout << i + 1 << ".rem" << ": " << squaredErrorSum / (SAMPLES % BATCH_SIZE) << std::endl; // mean squared error
				#endif
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
			inputs.push_back(Matrix::FromVector(in));

		std::vector<Matrix> targets;
		for (std::vector<double> tar : targetsVector)
			targets.push_back(Matrix::FromVector(tar));
		
		NeuralNetwork::batchTrain(inputs, targets, ITERATIONS, BATCH_SIZE);
	}

	/*
		TODO:
		Documentation
		maps each weight and bias in place
	*/
	void NeuralNetwork::mutate(doubleFunction func, double chance) {
		weights_ih.map(func, chance);

		for (int i = 0; i < weightsHidden.size(); i++)
				weightsHidden[i].map(func, chance);

		weights_ho.map(func, chance);

		if (chance == 1.0 || randomDouble(0, 1) < chance)
			bias_h0 = func(bias_h0);

		for (int i = 0; i < biasesHidden.size(); i++)
			if (chance == 1.0 || randomDouble(0, 1) < chance)
				biasesHidden[i] = func(biasesHidden[i]);

		if (chance == 1.0 || randomDouble(0, 1) < chance)
			bias_o = func(bias_o);
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