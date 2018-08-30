/*
	Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#pragma once

#ifdef TESTLIBRARY_EXPORTS  
#define TESTLIBRARY_API __declspec(dllexport)   
#else  
#define TESTLIBRARY_API __declspec(dllimport)   
#endif  

#include <vector>
#include <iostream>
#include <random>

namespace nn {

	class Matrix {
	private:
		std::vector<std::vector<double>> m;

		/*
			TODO:
			Documentation
			Matrix Addition
		*/
		Matrix add(Matrix b);

		/*
			TODO:
			Documentation
			Scalar Addition
		*/
		Matrix add(double b);

		/*
			TODO:
			Documentation
			Matrix Subtraction
		*/
		Matrix subtract(Matrix b);

		/*
			TODO:
			Documentation
			Scalar Subtraction
		*/
		Matrix subtract(double b);

		/*
			TODO:
			Documentation
			Hadamard product
		*/
		Matrix multiply(Matrix b);

		/*
			TODO:
			Documentation
			Scalar Multiplcation
		*/
		Matrix multiply(double b);

		/*
			TODO:
			Scalar Division
		*/
		Matrix divide(double b);

	public:
		TESTLIBRARY_API Matrix();

		TESTLIBRARY_API Matrix(std::vector<std::vector<double>> m);

		TESTLIBRARY_API Matrix(int rows, int columns, bool random = false);

#pragma region PrimaryOperations
		/*
			TODO:
			Documentation
			Matrix Addition
		*/
		TESTLIBRARY_API static Matrix Add(Matrix a, Matrix b);

		/*
			TODO:
			Documentation
			Scalar Addition
		*/
		TESTLIBRARY_API static Matrix Add(Matrix a, double b);

		/*
			TODO:
			Documentation
			Matrix Subtraction
		*/
		TESTLIBRARY_API static Matrix Subtract(Matrix a, Matrix b);

		/*
			TODO:
			Documentation
			Scalar Subtraction
		*/
		TESTLIBRARY_API static Matrix Subtract(Matrix a, double b);

		/*
			TODO:
			Documentation
			Hadamard Product
		*/
		TESTLIBRARY_API static Matrix Multiply(Matrix a, Matrix b);

		/*
			TODO:
			Documentation
			Scalar Multiplication
		*/
		TESTLIBRARY_API static Matrix Multiply(Matrix a, double b);

		/*
			Static Version
			TODO:
			Documentation
			Scalar Division
		*/
		TESTLIBRARY_API static Matrix Divide(Matrix a, double b);
#pragma endregion

		/*
			TODO:
			Documentation
			Dot Product
			produces a a.rows * b.columns matrix
		*/
		TESTLIBRARY_API static Matrix Dot(Matrix a, Matrix b);

		/*
			runs function against every element in the matrix
			optional param chance is the odds that the function will be ran (used for GA)
		*/
		TESTLIBRARY_API Matrix map(double(*func)(double), double chance = 1.0);

		/*
			runs function against every element in the matrix
			optional param chance is the odds that the function will be ran (used for GA)
		*/
		TESTLIBRARY_API static Matrix Map(Matrix a, double(*func)(double), double chance = 1.0);

		/*
			runs function against every element in the matrix
			function passes in a reference to this matrix along with it
			optional param chance is the odds that the function will be ran (used for GA)
		*/
		TESTLIBRARY_API Matrix map(double(*func)(double, Matrix), double chance = 1.0);

		/*
			runs function against every element in the matrix
			function passes in a reference to this matrix along with it
			optional param chance is the odds that the function will be ran (used for GA)
		*/
		TESTLIBRARY_API static Matrix Map(Matrix a, double(*func)(double, Matrix), double chance = 1.0);

		/*
			TODO:
			Documentation
			returns a transposed copy of the matrix
		*/
		TESTLIBRARY_API Matrix T();

		/*
			TODO:
			Documentation
			swaps rows and columns
		*/
		TESTLIBRARY_API Matrix transpose();

#pragma region Operators
		TESTLIBRARY_API bool operator== (const Matrix &rhs);
		TESTLIBRARY_API bool operator== (const std::vector<double> &rhs);
		TESTLIBRARY_API Matrix operator+ (const Matrix &rhs);
		TESTLIBRARY_API Matrix operator+ (const double &rhs);
		TESTLIBRARY_API Matrix& operator+= (const Matrix &rhs);
		TESTLIBRARY_API Matrix& operator+= (const double &rhs);
		TESTLIBRARY_API Matrix operator- (const Matrix &rhs);
		TESTLIBRARY_API Matrix operator- (const double &rhs);
		TESTLIBRARY_API Matrix& operator-= (const Matrix &rhs);
		TESTLIBRARY_API Matrix& operator-= (const double &rhs);
		TESTLIBRARY_API Matrix operator* (const Matrix &rhs);
		TESTLIBRARY_API Matrix operator* (const double &rhs);
		TESTLIBRARY_API Matrix& operator*= (const Matrix &rhs);
		TESTLIBRARY_API Matrix operator*= (const double &rhs);
		TESTLIBRARY_API Matrix operator/ (const double &rhs);
		TESTLIBRARY_API Matrix& operator/= (const double &rhs);
#pragma endregion
		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API double sum();

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API static Matrix FromVector(std::vector<double> a);

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API std::vector<double> toVector();

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API void randomise();

		/*
			TODO:
			Documentation
		*/
		TESTLIBRARY_API void print();

		/*
			TODO:
			Documentation
		*/
		static double randomDouble(double a, double b);
	};
}