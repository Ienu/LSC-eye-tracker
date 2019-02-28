# LSC eye-tracker (Learning Single Canera Eye-Tracker)
This repo is just for eye track research
## Category
### Deep Learning
Some models were collected here for training.
# Dataset
We are in the process of preparing dataset for free public release now.
# Codes
The code should follow Google style

# Eye Tracking Models
## i2g_g_v1.0.py
Basic deep learning model for eye tracking

Execute command should be like as follows, which can display and save log at the same time
```
python i2g_g_v1.0.py <datafile>.mat | tee <logfile>.txt
```
The <datafile>.mat should be saved as matlab v7.3 format

### dependencies
sklearn
