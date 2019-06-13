ROOT=$(pwd)

echo -e '\n\n--- install OS dependencies ---'
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

echo -e '\n\n--- install python packages ---'
pip install -r requirements.txt

if [ ! -d $ROOT/models ]; then
	echo -e '\n\n--- Clone tensorflow models ---'
	git clone git@github.com:tensorflow/models.git $ROOT/
fi

echo -e '\n\n--- Protobuf Compilation ---'
cd $ROOT/models/research/ && protoc object_detection/protos/*.proto --python_out=.

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/models/research/slim

echo -e '\n\n--- Testing the Installation ---'
cd $ROOT/models/research/ && python object_detection/builders/model_builder_test.py