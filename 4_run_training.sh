ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- Train pre-trained model ---'
cd $ROOT/tensorflow/models/research/ && python object_detection/model_main.py \
    --pipeline_config_path=$ROOT/data/flowers_faster_rcnn_resnet101.config \
    --model_dir=$ROOT/models \
    --alsologtostderr