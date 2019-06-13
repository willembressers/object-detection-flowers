# Object Detection of Flowers using TensorFlow API

Folder structure mainly based on [tensorflow/models/research/object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)

## 1. Install Tensorflow Object detection API
The Tensorflow Object Detection API has been installed as documented in the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). This includes installing library dependencies, compiling the configuration protobufs and setting up the Python environment.
```bash
./1_install_tensorflow.sh
```

## 2. Download training set
A valid data set has been created. See [this page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) for instructions on how to generate a dataset for the PASCAL VOC challenge or the Oxford-IIIT Pet dataset.
```bash
./2_download_dataset.sh
```

## 3. Configure object detecion pipline 
A Object Detection pipeline configuration has been written. See [this page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) for details on how to write a pipeline configuration.
```bash
./3_create_pipeline.sh
```

## 4. Running the Training Job
A local training job can be run with the following command:
```bash
./4_run_training.sh
```

## 5. Start the dashboard
Run the dash dashboard and manually upload (test) images.