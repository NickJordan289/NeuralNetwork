#pragma once

#ifdef TESTLIBRARY_EXPORTS  
#define TESTLIBRARY_API __declspec(dllexport)   
#else  
#define TESTLIBRARY_API __declspec(dllimport)   
#endif  

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace ml {
	namespace DataFuncs {
		/*
			Reads the given file seperating into lines and values
			seperates lines using the '\n' delimiter
			and seperates values using the given delim char (default ',')
		*/
		std::vector<std::vector<double>> loadFromFile(std::string filepath, char delim = ',') {
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
			Converts from an encoded label back to an integer
		*/
		int decodeOutput(std::vector<double> output) {
			double best;
			int bestIndex;
			for (int i = 0; i < output.size(); i++) {
				if (i == 0 || output[i] > best) {
					best = output[i];
					bestIndex = i;
				}
			}
			return bestIndex;
		}

		/*
			Splits a given vector into two vectors
			Split ratio is given by ratioA and ratioB
			default split ratio of 1/2
		*/
		template <typename T>
		void splitVector(std::vector<T> a, std::vector<T>& outB, std::vector<T>& outC, int ratioA = 1, int ratioB = 1) {
			int totalRatio = ratioA + ratioB;
			float percent = (float)ratioA / (float)totalRatio;
			for (int i = 0; i < ceil(a.size() * percent); i++)
				outB.push_back(a[i]);
			for (int i = ceil(a.size() * percent); i < a.size(); i++)
				outC.push_back(a[i]);
		}
	}
}