/*
	Author: Wenyu
	Date: 2/23/2019
	Version: 1.0
	Env: Opencv 3.4 vc14, VS2015 Release x64
	Function:
	v1.0: process gaze data and model an ANN from 12-D inputs to 2-D screen points
*/

#pragma once

#include <opencv2\opencv.hpp>

#define PI 3.1415926f

using namespace std;
using namespace cv;
using namespace cv::ml;

class GazeEst {
private:
	Ptr<ANN_MLP> m_network;
	const vector<float> x_scale = 
	{ 
		2, 2, 5, 
		PI, PI, PI, 
		640, 360, 
		1280, 720, 
		1280, 720 
	};
	const vector<float> y_scale = { 1920, 1080 };

public:
	GazeEst() {};

	// create model
	void create(){
		int N = 4;
		Mat layerSizes = (Mat_<int>(1, N) << 12, 25, 25, 2);
		m_network = ANN_MLP::create();
		m_network->setLayerSizes(layerSizes);
		m_network->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
		m_network->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, FLT_EPSILON));
		m_network->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.00001);	
	}

	// train model
	float train(const Mat& trainInputs, const Mat& trainOutputs) {
		// TODO: check dims
		// scale inputs and labels
		Mat trainData = Mat_<float>(trainInputs.rows, trainInputs.cols);
		Mat trainLabel = Mat_<float>(trainOutputs.rows, trainOutputs.cols);
		for (int i = 0; i < trainInputs.rows; ++i) {
			for (int j = 0; j < trainInputs.cols; ++j) {
				trainData.at<float>(i, j) = trainInputs.at<float>(i, j) / x_scale[j];
			}
			for (int k = 0; k < trainOutputs.cols; ++k) {
				trainLabel.at<float>(i, k) = trainOutputs.at<float>(i, k) / y_scale[k];
			}
		}

		// train
		Ptr<TrainData> tD = TrainData::create(
			trainData, 
			ROW_SAMPLE, 
			trainLabel);
		m_network->train(tD);

		// test
		Mat trainPredicts = Mat_<float>(trainInputs.rows, 2);
		predict(trainInputs, trainPredicts);
		Mat trainError;
		absdiff(trainOutputs, trainPredicts, trainError);
		Scalar s = mean(trainError);
		return float(s[0]);
	}

	// load model
	void load(const char* fileName) {
		m_network = ANN_MLP::load(fileName);
	}

	// save model
	void save(const char* fileName) {
		m_network->save(fileName);
	}

	// predict gaze point
	float predict(const Mat& testInputs, Mat& testOutputs, const Mat& testLabels = Mat()) {
		// TODO: check dims
		// scale inputs
		Mat testData = Mat_<float>(testInputs.rows, testInputs.cols);
		for (int i = 0; i < testInputs.rows; ++i) {
			for (int j = 0; j < testInputs.cols; ++j) {
				testData.at<float>(i, j) = testInputs.at<float>(i, j) / x_scale[j];
			}
		}
		// predict
		Mat predictLabel;
		m_network->predict(testData, predictLabel);
		// rescale outputs
		for (int i = 0; i < testInputs.rows; ++i) {
			for (int j = 0; j < 2; ++j) {
				testOutputs.at<float>(i, j) = predictLabel.at<float>(i, j) * y_scale[j];
			}
		}

		// test
		if (testLabels.rows == testOutputs.rows
			&& testLabels.cols == testOutputs.cols)
		{
			Mat testError;
			absdiff(testOutputs, testLabels, testError);
			Scalar s = mean(testError);
			return float(s[0]);
		}
		else {
			return 0;
		}
	}
};