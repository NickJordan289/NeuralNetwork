/*
Credit to Daniel Shiffman's videos on Perceptrons, Matrix Math and Neural Networks
*/

#include "Matrix.h"

namespace ml {
	Matrix::Matrix() {

	}

	/*
		Constructor that copies the contents of another
	*/
	Matrix::Matrix(std::vector<std::vector<double>> m) {
		this->m = m;
	}

	/*
		Standard Constructor
		Takes in rows, columns and if it should
		initialise with random values (def: true)
	*/
	Matrix::Matrix(int rows, int columns, bool random) {
		for (int i = 0; i < rows; i++) {
			std::vector<double> temp;
			for (int j = 0; j < columns; j++) {
				temp.push_back(0.f);
			}
			m.push_back(temp);
		}
		if (random)
			randomise();
	}

#pragma region PrimaryOperations
	/*
		Static Version
		TODO:
		Documentation
		Matrix Addition
	*/
	Matrix Matrix::Add(Matrix a, Matrix b) {
		Matrix result = Matrix(a);
		for (int row = 0; row < result.m.size(); row++)
			for (int col = 0; col < result.m[0].size(); col++)
				result.m[row][col] = a.m[row][col] + b.m[row][col];
		return result;
	}

	/*
		Static Version
		TODO:
		Documentation
		Scalar Addition
	*/
	Matrix Matrix::Add(Matrix a, double b) {
		return a.add(b);
	}

	/*
		TODO:
		Documentation
		Matrix Addition
	*/
	Matrix Matrix::add(Matrix b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[0].size(); col++)
				m[row][col] += b.m[row][col];
		return *this;
	}

	/*
		TODO:
		Documentation
		Scalar Addition
	*/
	Matrix Matrix::add(double b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				m[row][col] += b;
		return *this;
	}

	/*
		Static Version
		TODO:
		Documentation
		Matrix Subtraction
	*/
	Matrix Matrix::Subtract(Matrix a, Matrix b) {
		return a.subtract(b);
	}

	/*
		Static Version
		TODO:
		Documentation
		Scalar Subtraction
	*/
	Matrix Matrix::Subtract(Matrix a, double b) {
		return a.subtract(b);
	}

	/*
		TODO:
		Documentation
		Matrix Subtraction
	*/
	Matrix Matrix::subtract(Matrix b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				m[row][col] -= b.m[row][col];
		return *this;
	}

	/*
		TODO:
		Documentation
		Scalar Subtraction
	*/
	Matrix Matrix::subtract(double b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				m[row][col] -= b;
		return *this;
	}

	/*
		Static Version
		TODO:
		Documentation
		Hadamard Product
	*/
	Matrix Matrix::Multiply(Matrix a, Matrix b) {
		return a.multiply(b);
	}

	/*
		Static Version
		TODO:
		Documentation
		Scalar Multiplication
	*/
	Matrix Matrix::Multiply(Matrix a, double b) {
		return a.multiply(b);
	}

	/*
		TODO:
		Documentation
		Hadamard product
	*/
	Matrix Matrix::multiply(Matrix b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[0].size(); col++)
				m[row][col] *= b.m[row][col];
		return *this;
	}

	/*
		TODO:
		Documentation
		Scalar Multiplcation
	*/
	Matrix Matrix::multiply(double b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[0].size(); col++)
				m[row][col] *= b;
		return *this;
	}

	/*
		Static Version
		TODO:
		Documentation
		Scalar Division
	*/
	Matrix Matrix::Divide(Matrix a, double b) {
		return a.divide(b);
	}

	/*
		TODO:
		Documentation
		Scalar Division
	*/
	Matrix Matrix::divide(double b) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[0].size(); col++)
				m[row][col] /= b;
		return *this;
	}
#pragma endregion

	/*
		TODO:
		Documentation
		Dot Product
		produces a a.rows * b.columns matrix
	*/
	Matrix Matrix::Dot(Matrix a, Matrix b) {
		// Won't work if columns of A don't equal columns of B
		if (a.m[0].size() != b.m.size()) {
			throw std::invalid_argument("incompatible matrix sizes");
		}
		// Make a new matrix
		Matrix result = Matrix(a.m.size(), b.m[0].size(), false);
		for (int i = 0; i < a.m.size(); i++) {
			for (int j = 0; j < b.m[0].size(); j++) {
				// Sum all the rows of A times columns of B
				double sum = 0;
				for (int k = 0; k < a.m[0].size(); k++) {
					sum += a.m[i][k] * b.m[k][j];
				}
				// New value
				result.m[i][j] = sum;
			}
		}
		return result;
	}

	/*
		runs function against every element in the matrix
		optional param chance is the odds that the function will be ran (used for GA)
	*/
	Matrix Matrix::map(double(*func)(double), double chance) {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				if (chance == 1.0 || randomDouble(0, 1) < chance)
					m[row][col] = func(m[row][col]);
		return *this;
	}

	/*
		Static Version
		runs function against every element in the matrix
		optional param chance is the odds that the function will be ran (used for GA)
	*/
	Matrix Matrix::Map(Matrix a, double(*func)(double), double chance) {
		return a.map(func, chance);
	}

	/*
		runs function against every element in the matrix
		function passes in a reference to this matrix along with it
		optional param chance is the odds that the function will be ran (used for GA)
	*/
	Matrix Matrix::map(double(*func)(double, Matrix), double chance) {
		Matrix temp = *this;
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				if (chance == 1.0 || randomDouble(0, 1) < chance)
					temp.m[row][col] = func(temp.m[row][col], *this);
		*this = temp;
		return *this;
	}

	/*
		Static Version
		runs function against every element in the matrix
		function passes in a reference to this matrix along with it
		optional param chance is the odds that the function will be ran (used for GA)
	*/
	Matrix Matrix::Map(Matrix a, double(*func)(double, Matrix), double chance) {
		return a.map(func, chance);
	}

	/*
		TODO:
		Documentation
		returns a transposed copy of the matrix
	*/
	Matrix Matrix::T() {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix temp = *this;
		temp.transpose();
		return temp;
	}

	/*
		TODO:
		Documentation
		swaps rows and columns
	*/
	Matrix Matrix::transpose() {
		Matrix result = Matrix(m[0].size(), m.size(), false);
		for (int i = 0; i < result.m.size(); i++)
			for (int j = 0; j < result.m[0].size(); j++)
				result.m[i][j] = m[j][i];
		m = result.m;
		return *this;
	}

#pragma region Operators
	bool Matrix::operator== (const Matrix &rhs) {
		for (int i = 0; i < m.size(); i++)
			for (int j = 0; j < m[0].size(); j++)
				if (m[i][j] != rhs.m[i][j])
					return false;
		return true;
	}

	bool Matrix::operator== (const std::vector<double> &rhs) {
		Matrix temp = Matrix::FromVector(rhs);
		for (int i = 0; i < m.size(); i++)
			for (int j = 0; j < m[0].size(); j++)
				if (m[i][j] != temp.m[i][j])
					return false;
		return true;
	}

	Matrix Matrix::operator+ (const Matrix &rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.add(rhs);
		return lhs;
	}

	Matrix Matrix::operator+(const double & rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.add(rhs);
		return lhs;
	}

	Matrix& Matrix::operator+= (const Matrix &rhs) {
		this->add(rhs);
		return *this;
	}

	Matrix& Matrix::operator+= (const double &rhs) {
		this->add(rhs);
		return *this;
	}

	Matrix Matrix::operator- (const Matrix &rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.subtract(rhs);
		return lhs;
	}

	Matrix Matrix::operator-(const double & rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.subtract(rhs);
		return lhs;
	}

	Matrix& Matrix::operator-= (const Matrix &rhs) {
		this->subtract(rhs);
		return *this;
	}

	Matrix& Matrix::operator-=(const double & rhs) {
		this->subtract(rhs);
		return *this;
	}

	Matrix Matrix::operator* (const Matrix &rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.multiply(rhs);
		return lhs;
	}

	Matrix Matrix::operator* (const double &rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.multiply(rhs);
		return lhs;
	}

	Matrix& Matrix::operator*= (const Matrix &rhs) {
		this->multiply(rhs);
		return *this;
	}

	Matrix Matrix::operator*= (const double &rhs) {
		this->multiply(rhs);
		return *this;
	}

	Matrix Matrix::operator/ (const double &rhs) {
		// copy because we dont want to manipulate this object only use it for calculation
		Matrix lhs = Matrix(m);
		lhs.divide(rhs);
		return lhs;
	}

	Matrix& Matrix::operator/= (const double &rhs) {
		this->divide(rhs);
		return *this;
	}
#pragma endregion

	/*
		TODO:
		Documentation
	*/
	double Matrix::sum() {
		double total = 0.0;
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				total += m[row][col];
		return total;
	}

	/*
		TODO:
		Documentation
		if rows and cols are not defined it will create a 1 column matrix
	*/
	Matrix Matrix::FromVector(std::vector<double> a, int rows, int cols) {
		Matrix newMatrix = Matrix((rows == -1) ? a.size() : rows, cols, false);
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < cols; col++)
			newMatrix.m[row][col] = a[row*rows+col];
		return newMatrix;
	}

	/*
		TODO:
		Documentation
	*/
	std::vector<double> Matrix::toVector() {
		std::vector<double> temp;
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				temp.push_back(m[row][col]);
		return temp;
	}

	/*
		TODO:
		Documentation
	*/
	void Matrix::randomise() {
		for (int row = 0; row < m.size(); row++)
			for (int col = 0; col < m[row].size(); col++)
				m[row][col] = randomDouble(-1, 1);
	}

	/*
		TODO:
		Documentation
	*/
	void Matrix::print() {
		for (int row = 0; row < m.size(); row++) {
			std::cout << "[";
			for (int col = 0; col < m[row].size(); col++) {
				std::cout << m[row][col];
				if (col < m[row].size() - 1) {
					std::cout << ",";
				}
			}
			std::cout << "]";
			if (row < m.size() - 1) {
				std::cout << std::endl;
			}
		}
		std::cout << "\n" << std::endl;
	}

	/*
		TODO:
		Documentation
		This should be in its own file but I was lazy
	*/
	double randomDouble(double a, double b) {
		double random = ((double)rand()) / (double)RAND_MAX;
		double diff = b - a;
		double r = random * diff;
		return a + r;
	}
}