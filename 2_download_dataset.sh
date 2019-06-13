ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

if [ ! -d $ROOT/images ]; then
	echo -e '\n\n--- Extracting images ---'
	tar xf $ROOT/images.tar.gz -C $ROOT
fi

if [ ! -f $ROOT/data/flowers_train.record ]; then
	echo -e '\n\n--- Generate tf records ---'
	python generate_tf_records.py
fi

if [ ! -f $ROOT/data/model.ckpt.index ]; then
	echo -e '\n\n--- Downloading a COCO-pretrained Model for Transfer Learning ---'
	wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz -O $ROOT/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
	tar -xvf $ROOT/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
	cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* $ROOT/data/
	rm -rf $ROOT/faster_rcnn_resnet101_coco_11_06_2017.tar.gz $ROOT/faster_rcnn_resnet101_coco_11_06_2017
fi