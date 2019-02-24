/*
	Author: Wenyu
	Date: 2/24/2019
	Version: 1.1
	Env: Opencv 3.4 vc14, VS2015 Release x64
	Function:
	v1.0: process gaze data and model an ANN from 12-D inputs to 2-D screen points
	v1.1: add mat release, split cpp file, add destructor
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
	GazeEst();

	~GazeEst();

	// create model
	void create();

	// train model
	float train(const Mat&, const Mat&);

	// load model
	void load(const char*);

	// save model
	void save(const char*);

	// predict gaze point
	float predict(const Mat&, Mat&, const Mat& = Mat());
};
