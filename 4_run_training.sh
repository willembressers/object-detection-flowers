ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- Train pre-trained model ---'
PIPELINE_CONFIG_PATH=$ROOT/data/flowers_faster_rcnn_resnet101.config
MODEL_DIR=$ROOT/models
# NUM_TRAIN_STEPS=50000
# SAMPLE_1_OF_N_EVAL_EXAMPLES=1

cd $ROOT/tensorflow/models/research/ && python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr