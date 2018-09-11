#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include <memory>
#include <fstream>
#include <iterator>
#include <random>

#include <SFML/Graphics.hpp>

#include "NNLib.hpp"

#include "mnist_loader.h" // Credit https://github.com/arpaka/mnist-loader

/**
* http://www.tech-algorithm.com/articles/bilinear-image-scaling/
* Bilinear resize grayscale image.
* pixels is an array of size w * h.
* Target dimension is w2 * h2.
* w2 * h2 cannot be zero.
*
* @param pixels Image pixels.
* @param w Image width.
* @param h Image height.
* @param w2 New width.
* @param h2 New height.
* @return New array with size w2 * h2.
*/
std::vector<int> resizeBilinearGray(std::vector<int> pixels, int w, int h, int w2, int h2) {
	std::vector<int> temp;
	for (int i = 0; i < w2*h2; i++)
		temp.push_back(0);
	int A, B, C, D, x, y, index, gray;
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;
	float x_diff, y_diff, ya, yb;
	int offset = 0;
	for (int i = 0; i<h2; i++) {
		for (int j = 0; j<w2; j++) {
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			x_diff = (x_ratio * j) - x;
			y_diff = (y_ratio * i) - y;
			index = y * w + x;

			// range is 0 to 255 thus bitwise AND with 0xff
			A = pixels[index] & 0xff;
			B = pixels[index + 1] & 0xff;
			C = pixels[index + w] & 0xff;
			D = pixels[index + w + 1] & 0xff;

			// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
			gray = (int)(
				A*(1 - x_diff)*(1 - y_diff) + B * (x_diff)*(1 - y_diff) +
				C * (y_diff)*(1 - x_diff) + D * (x_diff*y_diff)
				);

			temp[offset++] = gray;
		}
	}
	return temp;
}

 int main() {
	srand(static_cast<unsigned int>(time(NULL)));

	const unsigned int SCREEN_WIDTH = 560;
	const unsigned int SCREEN_HEIGHT = 280;
	const unsigned int FPS = 0; //The desired FPS. (The number of updates each second) or 0 for uncapped.

	sf::Font ArialFont;
	ArialFont.loadFromFile("C:/Windows/Fonts/arial.ttf");

	sf::ContextSettings settings;
	settings.antialiasingLevel = 0;

	sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "MNIST", sf::Style::Default, settings);
	window.setFramerateLimit(FPS);
	window.setKeyRepeatEnabled(false);

#pragma region Graphics
	sf::Text lifetimeCounter;
	lifetimeCounter.setFont(ArialFont);
	lifetimeCounter.setString("0000");
	lifetimeCounter.setPosition(2, -4);
	lifetimeCounter.setOutlineColor(sf::Color::Black);
	lifetimeCounter.setOutlineThickness(1.f);

	sf::Text fpsCounter;
	fpsCounter.setFont(ArialFont);
	fpsCounter.setCharacterSize(20);
	fpsCounter.setString("0000");
	fpsCounter.setPosition(window.getSize().x - fpsCounter.getGlobalBounds().width - 6, 0);
	fpsCounter.setOutlineColor(sf::Color::Black);
	fpsCounter.setOutlineThickness(1.f);

	sf::Text correctText;
	correctText.setFont(ArialFont);
	correctText.setCharacterSize(20);
	correctText.setString("00000");
	correctText.setPosition(2*window.getSize().x/3, window.getSize().y - 20);
	correctText.setOutlineColor(sf::Color::Black);
	correctText.setOutlineThickness(1.f);

	sf::Text guessText;
	guessText.setFont(ArialFont);
	guessText.setCharacterSize(20);
	guessText.setString("0");
	guessText.setPosition(140, window.getSize().y-20);
	guessText.setOutlineColor(sf::Color::Black);
	guessText.setOutlineThickness(1.f);

	sf::RectangleShape leftBlock;
	leftBlock.setSize(sf::Vector2f(2.f, 280.f));
	leftBlock.setPosition(sf::Vector2f(window.getSize().x/2,0));

	sf::CircleShape brush;
	brush.setFillColor(sf::Color(255,255,255,200));
	brush.setRadius(4.f);
	brush.setOutlineThickness(4.f);
	brush.setOutlineColor(sf::Color(255,255,255,100));
	brush.setOrigin(brush.getGlobalBounds().width/2.f, brush.getGlobalBounds().height/2.f);
#pragma endregion

	sf::RenderTexture rt, rt2;
	rt.create(280, 280);
	rt2.create(280, 280);
	std::vector<sf::RectangleShape> rects, downsizedRects;

	mnist_loader train("train-images.idx3-ubyte",
		"train-labels.idx1-ubyte", 60000);

	mnist_loader test("t10k-images.idx3-ubyte",
		"t10k-labels.idx1-ubyte", 60000);

	int rows = train.rows();
	int cols = train.cols();

	ml::NeuralNetwork nn = ml::NeuralNetwork(784, {16, 16}, 10, 0.1);
	
	std::vector<int> pixels;

	int correct = 0;
	int guesses = 0;

	int i = 0;
	bool paused = false;
	sf::Event ev;
	sf::Clock clock;
	while (window.isOpen()) {
		float deltaTime = clock.restart().asSeconds();
		float fps = 1.f / deltaTime;
		fpsCounter.setString(std::to_string((int)fps));

		// Events
		while (window.pollEvent(ev)) {
			if (ev.type == sf::Event::Closed)
				window.close();

			if (ev.type == sf::Event::KeyPressed) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
					window.close();
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
					paused = !paused;
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1)) {
					window.setFramerateLimit(60); // unlimited framerate for faster generations
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2)) {
					window.setFramerateLimit(0); // unlimited framerate for faster generations
				}
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
					pixels.clear();
					downsizedRects.clear();
					rt.clear();
					rt2.clear();
				}
			}
		}

		// Update
		if (!paused) {
			if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && window.hasFocus()) {
				brush.setPosition(sf::Mouse::getPosition(window).x, sf::Mouse::getPosition(window).y);

				rt.draw(brush);

				sf::Image imm = rt.getTexture().copyToImage();
				imm.flipVertically();

				pixels.clear();

				for (int i = 0; i < rt.getSize().y; i++) {
					for (int j = 0; j < rt.getSize().x; j++) {
						pixels.push_back(imm.getPixel(j, i).r);
					}
				}

				pixels = resizeBilinearGray(pixels, 280, 280, 28, 28);

				double scale = 10;
				downsizedRects.clear();
				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {
						double brightness = (double)pixels[y * 28 + x];

						sf::RectangleShape r;
						r.setFillColor(sf::Color(brightness, brightness, brightness, 255));
						r.setSize(sf::Vector2f(scale, scale));
						r.setPosition(sf::Vector2f(x*scale, y*scale));
						downsizedRects.push_back(r);
					}
				}
			}
		}

		float scale = 10;

		rects.clear();
		int label = train.labels(i);
		std::vector<double> labelEnc = ml::DataFuncs::encodeLabel(label, 10);
		std::vector<double> image = train.images(i);

		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 28; x++) {
				image[y*28+x] = (double)sf::Uint8(image[y * 28 + x]*255.0)/255.0;
			}
		}

		nn.train(image, labelEnc);

		if (i % 100 == 0) {
			auto output = nn.predict(image).map(round);
			guesses++;
			if (output == labelEnc) {
				correct++;
			}
			correctText.setString(std::to_string(int(((double)correct / (double)guesses) * 100)) + "%");
		}

		if (pixels.size()>0) {
			std::vector<double> testPixels;
			for (int a : pixels) {
				testPixels.push_back((double)a/255.0);
			}
			auto output = nn.predict(testPixels);
			int digitGuess = ml::DataFuncs::decodeOutput(output.toVector());
			guessText.setString(std::to_string(digitGuess));
		}

		if (i < 59999)
			i++;
		else {
			i = 0;
			int randomSeed = static_cast<unsigned int>(time(NULL));
			std::cout << "Shuffle! " << randomSeed << std::endl;
			srand(randomSeed);
			std::random_shuffle(std::begin(train.m_images), std::end(train.m_images));
			srand(randomSeed);
			std::random_shuffle(std::begin(train.m_labels), std::end(train.m_labels));
			
			correct = 0;
			guesses = 0;
		}
		
		for (auto a : downsizedRects) {
			rt2.draw(a);
		}
		downsizedRects.clear();

		// Draw
		window.clear();

		for (auto a : rects) {
			window.draw(a);
		}

		window.draw(leftBlock);

		sf::Image im = rt.getTexture().copyToImage();
		im.flipVertically();
		sf::Texture t;
		t.loadFromImage(im);
		sf::Sprite s;
		s.setTexture(t);
		window.draw(s);

		window.draw(fpsCounter);
		window.draw(correctText);
		window.draw(guessText);

		window.display();
	}

	return 0;
}