/*
	Author: Wenyu
	Date: 2/28/2019
	Version: 1.5
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
*/

#pragma once

#include <opencv2\opencv.hpp>

namespace ge {
	class GazeEst {
	private:
		cv::Ptr<cv::ml::ANN_MLP> m_network;
		const float PI = 3.1415926f;
		const float x_scale[12] =
		{
			2, 2, 5,
			PI, PI, PI,
			640, 360,
			1280, 720,
			1280, 720
		};
		const float y_scale[2] = { 1920, 1080 };

		double m_train_time;
		double m_pre_time;

	public:

		/** @init the model for ANN */
		GazeEst();

		/** @destruct the model and release memory */
		~GazeEst();

		/** @breif create the model */
		void create();

		/** @train the model with specific params, output the training loss

		@param trainInputs: samples for training, each row indicates one sample
		@param trainOutputs: sample labels for training, each row indicates one sample,	the
			number of rows should equal to the training samples
		@param stop_error: stop error for training, when the error is less than the stop error,
			the process halts
		@param testInputs: samples for testing, ecah row indicates one sample
		@param testOuputs: sample labels for training, each row indicates one sample, the number
			of rows should equal to the testing samples
		@param verbose: true for display training process
		 */
		float train(const cv::Mat& trainInputs,
			const cv::Mat& trainOutputs,
			float stop_error = -1,
			const cv::Mat& testInputs = cv::Mat(),
			const cv::Mat& testOutputs = cv::Mat(),
			bool verbose = false);

		/** @get the training time */
		double getTrainTime();

		/** @load pre-trained xml model file

		@param fileName: xml file path
		 */
		void load(const char* fileName);

		/** @save trained xml model file

		@param fileName: xml file path
		 */
		void save(const char* fileName);

		/** @predict labels from test samples, output the test loss

		@param testInputs: test samples, each row indicates one sample
		@param testOutputs: output the predicted labels
		@param testLabels: labels for test samples, can be used to computed the test error
		 */
		float predict(const cv::Mat& testInputs,
			cv::Mat& testOutputs,
			const cv::Mat& testLabels = cv::Mat());

		/** @get the testing time */
		double getTestTime();

		/** @static method for shuffling the data

		@param src: original samples
		@param dst: shuffled samples
		 */
		static void shuffle(const cv::Mat& src, cv::Mat& dst);
	};

}