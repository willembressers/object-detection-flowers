ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

echo -e '\n\n--- install OS dependencies ---'
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

echo -e '\n\n--- install python packages ---'
pip install -r requirements.txt

if [ ! -d $ROOT/tensorflow/models ]; then
	echo -e '\n\n--- Clone tensorflow models ---'
	git clone git@github.com:tensorflow/models.git $ROOT/tensorflow/models
fi

if [ ! -d $ROOT/cocoapi ]; then
	echo -e '\n\n--- Clone cocoapi ---'
	git clone https://github.com/cocodataset/cocoapi.git $ROOT/cocoapi
	cd $ROOT/cocoapi/PythonAPI
	make
	cp -r pycocotools $ROOT/tensorflow/models/research/
fi

echo -e '\n\n--- Protobuf Compilation ---'
cd $ROOT/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- Testing the Installation ---'
cd $ROOT/tensorflow/models/research/ && python object_detection/builders/model_builder_test.py