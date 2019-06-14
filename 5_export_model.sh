ROOT=$(pwd)
CHECKPOINT_NUMBER=1014

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- Exporting the Tensorflow Graph ---'
mkdir -p $ROOT/output/$CHECKPOINT_NUMBER

cd $ROOT/tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $ROOT/data/flowers_faster_rcnn_resnet101.config \
    --trained_checkpoint_prefix $ROOT/models/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory $ROOT/output/$CHECKPOINT_NUMBER