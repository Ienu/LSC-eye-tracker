/*
	Author: Wenyu
	Date: 03/04/2019
	Version: 1.7
	Env: Opencv 3.4 vc14, VS2015 Release x64
	Function:
	v1.0: process gaze data and model an ANN from 12-D inputs to 2-D screen points
	v1.1: add mat release, split cpp file, add destructor
	v1.2: add static shuffle function for data preprocessing, add time consumption
	analysis, change the model
	v1.3: adjust the model, add comments, add function to stop training with a specific
	loss
	v1.4: change opt method to rprop
	v1.5: improve code with namespace and add adaptive training stop
	v1.6: add incrementally training method
	v1.7: add visualize method
*/

#include "gazeEstimate.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cassert>

ge::GazeEst::GazeEst() {
	m_train_time = -1;
	m_pre_time = -1;
	m_n_samples = -1;
}

ge::GazeEst::~GazeEst() {
	m_network.release();
}

void ge::GazeEst::create() {
	int N = 6;
	cv::Mat layerSizes = (cv::Mat_<int>(1, N) << 12, 50, 50, 25, 12, 2);
	m_network = cv::ml::ANN_MLP::create();
	m_network->setLayerSizes(layerSizes);
	m_network->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	m_network->setTermCriteria(
		cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 
			1, 1e-9/*cv::FLT_EPSILON*/));
	m_network->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.01, 0.01);
}

float ge::GazeEst::train(
	const cv::Mat& trainInputs, 
	const cv::Mat& trainOutputs, 
	float stopError,
	const cv::Mat& testInputs, 
	const cv::Mat& testOutputs, 
	bool verbose) 
{
	assert(trainInputs.rows > 0
		&& trainInputs.rows == trainOutputs.rows
		&& trainInputs.cols == 12
		&& trainOutputs.cols == 2);

	// scale inputs and labels
	std::srand((unsigned int)time(NULL));
	cv::Mat trainData = cv::Mat_<float>(trainInputs.rows, trainInputs.cols);
	cv::Mat trainLabel = cv::Mat_<float>(trainOutputs.rows, trainOutputs.cols);
	for (int i = 0; i < trainInputs.rows; ++i) {
		for (int j = 0; j < trainInputs.cols; ++j) {
			trainData.at<float>(i, j) = trainInputs.at<float>(i, j) / x_scale[j];
		}
		for (int k = 0; k < trainOutputs.cols; ++k) {
			trainLabel.at<float>(i, k) = trainOutputs.at<float>(i, k) / y_scale[k];
		}
	}
	m_n_samples += trainData.rows;

	// train
	cv::Ptr<cv::ml::TrainData> tD = cv::ml::TrainData::create(
		trainData,
		cv::ml::ROW_SAMPLE,
		trainLabel);
	cv::Mat trainPredicts = cv::Mat_<float>(trainInputs.rows, 2);
	cv::Mat trainError, pre;
	cv::Scalar s;
	m_network->train(tD);
	std::ofstream f_train("train_loss.txt");

	std::vector<float> t_error;
	std::vector<float> t_avg_error;

	std::vector<float> p_error;
	std::vector<float> p_avg_error;

	std::time_t t_start = std::clock();
	for (int epoch = 0; epoch < 50000; epoch++) {
		std::time_t start = std::clock();
		m_network->train(tD, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
		double train_time = double(std::clock() - start) / CLOCKS_PER_SEC;

		// test
		float ts = -1;
		if (testInputs.rows == testOutputs.rows && testInputs.rows != 0) {
			assert(testInputs.cols == 12 && testOutputs.cols == 2);
			ts = predict(testInputs, pre, testOutputs);
		}
		
		predict(trainInputs, trainPredicts);
		cv::absdiff(trainOutputs, trainPredicts, trainError);
		s = cv::mean(trainError);

		// stop condition
		if (ts < stopError && stopError != -1) {
			break;
		}
		// adaptive training
		else if (stopError == -1) {
			const int e_win = 10;
			t_error.push_back(float(s[0]));
			p_error.push_back(ts);
			if (epoch >= e_win - 1) {
				float tae = 0;
				float pae = 0;
				for (int eIdx = epoch - e_win + 1; eIdx <= epoch; ++eIdx) {
					tae += t_error[eIdx];
					pae += p_error[eIdx];
				}
				t_avg_error.push_back(tae / e_win);
				p_avg_error.push_back(pae / e_win);
			}
			if (epoch >= e_win * 10 && (p_avg_error.back() >= t_avg_error.back() * 1.01
				|| t_avg_error.back() >= t_avg_error[t_avg_error.size() - 2] * 1.0001)) {
				break;
			}
		}

		if (verbose)
		{
			std::cout << epoch << ":\tTrain Loss: " << s[0] 
				<< "\tTest Loss: " << ts << "\ttime: " << train_time << std::endl;
		}
		
		f_train << epoch << "\t" << s[0] << "\t" << ts << std::endl;
	}
	
	m_train_time = double(std::clock() - t_start) / CLOCKS_PER_SEC;
	f_train.close();

	trainPredicts.release();
	trainError.release();

	trainData.release();
	trainLabel.release();

	pre.release();
	return float(s[0]);
}


float ge::GazeEst::incTrain(
	const cv::Mat& trainInputs,
	const cv::Mat& trainOutputs,
	const cv::Mat& testInputs,
	const cv::Mat& testOutputs,
	bool verbose) 
{
	assert(trainInputs.rows > 0
		&& trainInputs.rows == trainOutputs.rows
		&& trainInputs.cols == 12
		&& trainOutputs.cols == 2);

	// scale inputs and labels
	cv::Mat trainData = cv::Mat_<float>(trainInputs.rows, trainInputs.cols);
	cv::Mat trainLabel = cv::Mat_<float>(trainOutputs.rows, trainOutputs.cols);
	for (int i = 0; i < trainInputs.rows; ++i) {
		for (int j = 0; j < trainInputs.cols; ++j) {
			trainData.at<float>(i, j) = trainInputs.at<float>(i, j) / x_scale[j];
		}
		for (int k = 0; k < trainOutputs.cols; ++k) {
			trainLabel.at<float>(i, k) = trainOutputs.at<float>(i, k) / y_scale[k];
		}
	}

	// train
	cv::Ptr<cv::ml::TrainData> tD = cv::ml::TrainData::create(
		trainData,
		cv::ml::ROW_SAMPLE,
		trainLabel);
	cv::Mat trainPredicts = cv::Mat_<float>(trainInputs.rows, 2);
	cv::Mat trainError, pre;
	cv::Scalar s;

	std::vector<float> t_error;
	std::vector<float> t_avg_error;

	std::vector<float> p_error;
	std::vector<float> p_avg_error;

	// compute train epochs
	int train_epochs = 1;
	if (m_n_samples > 0) {
		int n = trainData.rows * 10 / m_n_samples;
		train_epochs = n > 1 ? n : 1;
	}
	
	// incrementally train
	std::time_t t_start = std::clock();
	for (int epoch = 0; epoch < train_epochs; epoch++) {
		std::time_t start = std::clock();
		m_network->train(tD, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
		double train_time = double(std::clock() - start) / CLOCKS_PER_SEC;

		// test
		float ts = -1;
		if (testInputs.rows == testOutputs.rows && testInputs.rows != 0) {
			assert(testInputs.cols == 12 && testOutputs.cols == 2);
			ts = predict(testInputs, pre, testOutputs);
		}

		predict(trainInputs, trainPredicts);
		cv::absdiff(trainOutputs, trainPredicts, trainError);
		s = cv::mean(trainError);

		if (verbose) {
			std::cout << epoch << ":\tTrain Loss: " << s[0]
				<< "\tTest Loss: " << ts << "\ttime: " << train_time << std::endl;
		}
	}

	m_inc_train_time = double(std::clock() - t_start) / CLOCKS_PER_SEC;

	m_n_samples += trainData.rows;

	trainPredicts.release();
	trainError.release();

	trainData.release();
	trainLabel.release();

	pre.release();
	return float(s[0]);
}

double ge::GazeEst::getTrainTime() {
	return m_train_time;
}

double ge::GazeEst::getIncTrainTime() {
	return m_inc_train_time;
}

unsigned int ge::GazeEst::getNumTrained() {
	return m_n_samples;
}

void ge::GazeEst::load(const char* fileName, int nSamples) {
	m_network = cv::ml::ANN_MLP::load(fileName);
	m_n_samples = nSamples;
}

void ge::GazeEst::save(const char* fileName) {
	m_network->save(fileName);
}

float ge::GazeEst::predict(
	const cv::Mat& testInputs, 
	cv::Mat& testOutputs, 
	const cv::Mat& testLabels) {
	assert(testInputs.rows == testOutputs.rows 
		&& testInputs.cols == 12 
		&& testOutputs.cols == 2);

	// scale inputs
	cv::Mat testData = cv::Mat_<float>(testInputs.rows, testInputs.cols);
	for (int i = 0; i < testInputs.rows; ++i) {
		for (int j = 0; j < testInputs.cols; ++j) {
			testData.at<float>(i, j) = testInputs.at<float>(i, j) / x_scale[j];
		}
	}
	// predict
	cv::Mat predictLabel;

	std::time_t t_start = std::clock();
	m_network->predict(testData, predictLabel);
	m_pre_time = double(std::clock() - t_start) / CLOCKS_PER_SEC / testData.rows;

	// rescale outputs
	testOutputs = cv::Mat_<float>(testInputs.rows, 2);
	for (int i = 0; i < testInputs.rows; ++i) {
		for (int j = 0; j < 2; ++j) {
			testOutputs.at<float>(i, j) = predictLabel.at<float>(i, j) * y_scale[j];
		}
	}

	testData.release();
	predictLabel.release();

	// test
	if (testLabels.rows == testOutputs.rows
		&& testLabels.cols == testOutputs.cols)
	{
		cv::Mat testError;
		cv::absdiff(testOutputs, testLabels, testError);
		cv::Scalar s = cv::mean(testError);
		testError.release();
		return float(s[0]);
	}
	else {
		return -1;
	}
}

double ge::GazeEst::getTestTime() {
	return m_pre_time;
}

void ge::GazeEst::shuffle(const cv::Mat& src, cv::Mat& dst) {
	// index not exceed 65535
	int n = src.rows;
	cv::Mat l_dst= cv::Mat_<float>(src.rows, src.cols);

	std::srand((unsigned int)time(NULL));
	int* sIdx = new int[n];

	// init sequence
	for (int i = 0; i < n; ++i) {
		sIdx[i] = i;
	}

	// shuffle sequence index
	for (int j = 0; j < n; j++) {
		int r = rand() % (n - j);
		int tp = sIdx[r];
		sIdx[r] = sIdx[n - j - 1];
		sIdx[n - j - 1] = tp;
	}

	// reorder the data
	for (int k = 0; k < n; ++k) {
		for (int l = 0; l < src.cols; ++l) {
			l_dst.at<float>(k, l) = src.at<float>(sIdx[k], l);
		}
	}
	dst = l_dst.clone();
	l_dst.release();
	delete[] sIdx;
}

void ge::GazeEst::visualize(const cv::Mat& testLabel, const cv::Mat& predictLabel, int width, int height)
{
	assert(testLabel.rows == predictLabel.rows && testLabel.cols == 2 && predictLabel.cols == 2);
	cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < testLabel.rows; ++i) {
		int tx = int(testLabel.at<float>(i, 0));
		int ty = int(testLabel.at<float>(i, 1));
		cv::circle(image, cv::Point(tx, ty), 2, cv::Scalar(0, 0, 255), 2);

		int px = int(predictLabel.at<float>(i, 0));
		int py = int(predictLabel.at<float>(i, 1));
		cv::circle(image, cv::Point(px, py), 2, cv::Scalar(255, 0, 0), 2);

		cv::line(image, cv::Point(tx, ty), cv::Point(px, py), cv::Scalar(0, 0, 0), 1);
	}
	cv::imshow("Visualization", image);
	cv::waitKey(0);
}
