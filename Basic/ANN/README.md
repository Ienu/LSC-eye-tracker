# Gaze Estimation
We use conventional algorithms to obtain the feature vector, then use ANN proposed in this repo to regress to the screen gaze point
## Namespace
### ge
## Class
### ge::GazeEst
## Functions
### GazeEst
```
GazeEst()
```
init the model for ANN
### ~GazeEst
```
~GazeEst()
```
destruct the model and release memory
### create
```
void create()
```
breif create the model
### train
```
float train(const cv::Mat& trainInputs, 
            const cv::Mat& trainOutputs, 
            float stopError = -1, 
            const cv::Mat& testInputs = cv::Mat(), 
            const cv::Mat& testOutputs = cv::Mat(), 
            bool verbose = false)
```
train the model with specific params, output the training loss
* trainInputs: samples for training, each row indicates one sample
* trainOutputs: sample labels for training, each row indicates one sample, the number of rows should equal to the training samples
* stop_error: stop error for training, when the error is less than the stop error, the process halts, default value is -1, it will stop adaptively
* testInputs: samples for testing, ecah row indicates one sample
* testOuputs: sample labels for training, each row indicates one sample, the number of rows should equal to the testing samples
* verbose: true for display training process
### incTrain
```
float incTrain(const cv::Mat& trainInputs, 
               const cv::Mat& trainOutputs, 
               const cv::Mat& testInputs = cv::Mat(), 
               const cv::Mat& testOutputs = cv::Mat(), 
               bool verbose = false)
```
incrementally train the model with specific params, output the training loss
* trainInputs: samples for training, each row indicates one sample
* trainOutputs: sample labels for training, each row indicates one sample, the number of rows should equal to the training samples
* testInputs: samples for testing, ecah row indicates one sample
* testOuputs: sample labels for training, each row indicates one sample, the number of rows should equal to the testing samples
* verbose: true for display training process
### getTrainTime
```
double getTrainTime()
```
get the training time
### getIncTrainTime
```
double getIncTrainTime()
```
get the incremental training time
### getNumTrained
```
unsigned int getNumTrained()
```
get the number of trained samples in total
### load
```
void load(const char* fileName, 
          int nSamples = -1)
```
load pre-trained xml model file
* fileName: xml file path
* nSamples: previous training samples amount, for incremental train, default value is -1, which makes the incTrain learn the new samples once
### save
```
void save(const char* fileName)
```
save trained xml model file
* fileName: xml file path
### predict
```
float predict(const cv::Mat& testInputs, 
              cv::Mat& testOutputs, 
              const cv::Mat& testLabels = cv::Mat())
```
predict labels from test samples, output the test loss
* testInputs: test samples, each row indicates one sample
* testOutputs: output the predicted labels
* testLabels: labels for test samples, can be used to computed the test error
### getTestTime
```
double getTestTime()
```
get the testing time
### shuffle
```
static void shuffle(const cv::Mat& src, 
                    cv::Mat& dst)	
```
static method for shuffling the data
* src: original samples
* dst: shuffled samples
