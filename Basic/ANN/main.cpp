/*
	Author: Wenyu
	Date: 2/24/2019
	Version: 1.1
	Env: Opencv 3.4 vc14, VS2015 Release x64, "gazeEstimate.h", "gazeEstimate.cpp"
	Function:
	v1.0: process gaze data and model an ANN from 12-D inputs to 2-D screen points
		as a demo
	v1.1: add mat release
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2\opencv.hpp>

#include "gazeEstimate.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main() {
	// import data
	const int amount = 400;
	ifstream f_data("data.txt");
	Mat MD = Mat_<float>(amount, 14);
	for (int index = 0; index < amount; ++index) {
		for (int i = 0; i < 14; ++i) {
			float tp;
			f_data >> tp;
			MD.at<float>(index, i) = tp;
		}
	}
	f_data.close();

	// split data into train set and test set
	int nTrain = amount * 4 / 5;
	Mat trainInputs = MD(Rect(0, 0, 12, nTrain));
	Mat trainOutputs = MD(Rect(12, 0, 2, nTrain));

	int nTest = amount - nTrain;
	Mat testInputs = MD(Rect(0, nTrain, 12, nTest));
	Mat testOutputs = MD(Rect(12, nTrain, 2, nTest));

	// train n test
	GazeEst gE;
	gE.create();
	
	float trainLoss = gE.train(trainInputs, trainOutputs);
	cout << "trainLoss = " << trainLoss << endl;

	// load model
	// gE.load("test.xml");
	Mat pre = Mat_<float>(nTest, 2);
	float testLoss = gE.predict(testInputs, pre, testOutputs);
	cout << "testLoss = " << testLoss << endl;

	cout << pre << endl;
	
	// save model
	gE.save("test.xml");

	MD.release();
	pre.release();
	return 0;
}
