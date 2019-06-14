import os
import click
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

# get the full path to the root of this repo
root_dir = os.path.abspath(os.path.sep.join([__file__, os.pardir]))

# append the path to the system path so python can load the research modules
import sys
sys.path.append(os.path.sep.join([root_dir, 'tensorflow', 'models', 'research']))

# Import the tensorflow/models/research/ modules AFTER the path append
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

graph_path = os.path.sep.join([root_dir, 'output', 'model', 'frozen_inference_graph.pb'])
labels_path = os.path.sep.join([root_dir, 'data', 'flowers_label_map.pbtxt'])

# Load the tensorflow graph
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()

	with tf.gfile.GFile(graph_path, 'rb') as fid:
		serialized_graph = fid.read()

		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# load the labels
category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size

	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:

			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}

			for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
		
			if 'detection_masks' in tensor_dict:
			
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
	
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
				detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
	
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
			
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
			
			return output_dict


def inference(image, min_score_thresh=.5):
	objects = []

	(im_width, im_height) = image.size

	# the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
	image_np = load_image_into_numpy_array(image)
	
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)

	# Actual detection.
	output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

	num_detections = output_dict['num_detections']
	classes = output_dict['detection_classes']
	boxes = output_dict['detection_boxes']
	scores = output_dict['detection_scores']

	for i in range(num_detections):
		if scores is None or scores[i] > min_score_thresh:
			if classes[i] in category_index.keys():
				ymin, xmin, ymax, xmax = boxes[i]
				(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

				objects.append({
					'confidence':int(100*scores[i]),
					'class':category_index[classes[i]]['name'],
					'image':image.crop((left, top, right, bottom)),
				})

	return objects
