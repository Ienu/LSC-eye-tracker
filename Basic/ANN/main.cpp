/*
	Author: Wenyu
	Date: 2/28/2019
	Version: 1.2
	Env: Opencv 3.4 vc14, VS2015 Release x64, "gazeEstimate.h", "gazeEstimate.cpp"
	Function:
	v1.0: process gaze data and model an ANN from 12-D inputs to 2-D screen points
		as a demo
	v1.1: add mat release
	v1.2: use namespace ge
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2\opencv.hpp>

#include "gazeEstimate.h"

int main() {
	// import data
	const int amount = 5388;
	std::ifstream f_data("data7.txt");
	cv::Mat MD = cv::Mat_<float>(amount, 14);
	for (int index = 0; index < amount; ++index) {
		for (int i = 0; i < 14; ++i) {
			float tp;
			f_data >> tp;
			MD.at<float>(index, i) = tp;
		}
	}
	f_data.close();

	ge::GazeEst::shuffle(MD, MD);

	// split data into train set and test set
	int nTrain = amount * 4 / 5;
	cv::Mat trainInputs = MD(cv::Rect(0, 0, 12, nTrain));
	cv::Mat trainOutputs = MD(cv::Rect(12, 0, 2, nTrain));

	int nTest = amount - nTrain;
	cv::Mat testInputs = MD(cv::Rect(0, nTrain, 12, nTest));
	cv::Mat testOutputs = MD(cv::Rect(12, nTrain, 2, nTest));

	// train n test
	ge::GazeEst gE;
	gE.create();
	
	float trainLoss = gE.train(trainInputs, trainOutputs, -1, testInputs, testOutputs, true);
	std::cout << "trainLoss = " << trainLoss << "\ttime = " << gE.getTrainTime() << std::endl;


	// load model
	//gE.load("total.xml");
	cv::Mat pre = cv::Mat_<float>(nTest, 2);
	float testLoss = gE.predict(testInputs, pre, testOutputs);
	std::cout << "testLoss = " << testLoss << "\ttime = " << gE.getTestTime() << std::endl;

	//cout << pre << endl;
	
	// save model
	gE.save("latest_model_6e70.xml");

	MD.release();
	pre.release();
	return 0;
}