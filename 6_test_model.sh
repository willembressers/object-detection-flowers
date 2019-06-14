ROOT=$(pwd)

# SEE https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/tensorflow/models/research/slim

echo -e '\n\n--- TEST ---'
python test_model.py --graph output/1014/frozen_inference_graph.pb --labels data/flowers_label_map.pbtxt