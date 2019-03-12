# Visualization codes for LSC-eye-tracker

## Model Visualization

### Layer visualization

`python layer-visualization.py <root_directory_of_npz_file> <root_directory_of_model>`

### Kernel visualization

#### not finished!

`python kerne-visualization.py <path_to_model>`

### Image deconv

Only use the first layer of model

`python image_deconv.py <root_directory_of_npz_file> <root_directory_of_model>`

## Generate dataset

version-4.2 can generate dataset of every person

version-4.3 can generete dateset of all person into one npz file

`python generate-dataset-v4.2(3).py <root_directory_of_dataset>`