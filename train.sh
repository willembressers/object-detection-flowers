ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

if [ ! -f $ROOT/data/train.record ]; then
	echo -e '\n\n--- Generate tf records ---'
	python src/generate_tf_records.py
fi 

echo -e '\n\n--- Add Libraries to PYTHONPATH ---'
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- Train pre-trained model ---'
PIPELINE_CONFIG_PATH=$ROOT/models/ssd_inception_v2_coco.config
MODEL_DIR=$ROOT/models/ssd_inception_v2_coco_2017_11_17
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

cd $ROOT/tensorflow/models/research/ && python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr