ROOT=$(pwd)

echo -e '\n\n--- install OS dependencies ---'
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

echo -e '\n\n--- install python packages ---'
pip install -r requirements.txt

if [ ! -d $ROOT/src/models ]; then
	echo -e '\n\n--- Clone tensorflow models ---'
	git clone git@github.com:tensorflow/models.git $ROOT/src/models
fi

echo -e '\n\n--- Protobuf Compilation ---'
cd $ROOT/src/models/research/ && protoc object_detection/protos/*.proto --python_out=.

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src/models/research/slim

echo -e '\n\n--- Testing the Installation ---'
cd $ROOT/src/models/research/ && python object_detection/builders/model_builder_test.py

if [ ! -d $ROOT/data/images ]; then
	echo -e '\n\n--- Extracting images ---'
	tar xf $ROOT/data/images.tar.gz -C $ROOT/data
fi