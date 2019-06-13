ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

if [ ! -f $ROOT/data/flowers_faster_rcnn_resnet101.config ]; then
	echo -e '\n\n--- Configuring the Object Detection Pipeline ---'
	cd $ROOT/tensorflow/models/research/ && cp object_detection/samples/configs/faster_rcnn_resnet101_pets.config $ROOT/data/flowers_faster_rcnn_resnet101.config
	sed -i "s|PATH_TO_BE_CONFIGURED|"$ROOT"/data|g" $ROOT/data/flowers_faster_rcnn_resnet101.config
	sed -i "s|pet_faces|flowers|g" $ROOT/data/flowers_faster_rcnn_resnet101.config
	sed -i "s|pet_label_map|flowers_label_map|g" $ROOT/data/flowers_faster_rcnn_resnet101.config
	sed -i "s|record-?????-of-00010|record|g" $ROOT/data/flowers_faster_rcnn_resnet101.config
fi 