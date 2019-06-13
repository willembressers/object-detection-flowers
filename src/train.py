import io
import os
import click
import pandas
from PIL import Image
import tensorflow as tf
import xml.etree.ElementTree as ET

# get the full path to the root of this repo
root_dir = os.path.abspath(os.path.sep.join([__file__, os.pardir, os.pardir]))


def xml_2_df(images_directory):

	anchors = []

	# loop over the given root directory
	for sub_directory in os.listdir(images_directory):

		# get the path of the sub directory
		path = os.path.sep.join([images_directory, sub_directory])

		# is the sub directory 
		if os.path.isdir(path):

			# loop over the files in the sub directory
			for file_name in os.listdir(path):

				# only work with xml files
				if file_name.endswith(".xml"): 

					# get the root of the XML tree
					root = ET.parse(os.path.sep.join([path, file_name])).getroot()

					# loop over all objects
					for member in root.findall('object'):

						# collect the information
						anchors.append({
							'stage':sub_directory,
							'file_path':os.path.sep.join([path, root.find('filename').text]),
							'width':int(root.find('size')[0].text),
							'height':int(root.find('size')[1].text),
							'class':member[0].text,
							'xmin':int(member[4][0].text),
							'ymin':int(member[4][1].text),
							'xmax':int(member[4][2].text),
							'ymax':int(member[4][3].text)
						})
						

	return pandas.DataFrame(anchors)

def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def aggregate_file_group(file_group, width, height, classes):
	xmins = []
	xmaxs = []
	ymins = []
	ymaxs = []
	text = []
	label = []

	for index, row in file_group.iterrows():
		xmins.append(row['xmin'] / width)
		xmaxs.append(row['xmax'] / width)
		ymins.append(row['ymin'] / height)
		ymaxs.append(row['ymax'] / height)
		text.append(row['class'].encode('utf8'))
		label.append(classes[row['class']])
		
	return xmins, xmaxs, ymins, ymaxs, text, label


def create_tf_example(file_path, file_group, classes):
	# read the file (with tensorflow)
	with tf.gfile.GFile(file_path, 'rb') as fid:			
		encoded_jpg = fid.read()
		encoded_jpg_io = io.BytesIO(encoded_jpg)

		# read image with PIL 
		image = Image.open(encoded_jpg_io)

		# get image dimensions
		width, height = image.size

		# aggregate the file group
		xmins, xmaxs, ymins, ymaxs, texts, labels =  aggregate_file_group(file_group, width, height, classes)

		# create the example
		return tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(height),
			'image/width': int64_feature(width),
			'image/filename': bytes_feature(file_path.encode('utf8')),
			'image/source_id': bytes_feature(file_path.encode('utf8')),
			'image/encoded': bytes_feature(encoded_jpg),
			'image/format': bytes_feature( b'jpg'),
			'image/object/bbox/xmin': float_list_feature(xmins),
			'image/object/bbox/xmax': float_list_feature(xmaxs),
			'image/object/bbox/ymin': float_list_feature(ymins),
			'image/object/bbox/ymax': float_list_feature(ymaxs),
			'image/object/class/text': bytes_list_feature(texts),
			'image/object/class/label': int64_list_feature(labels),
		}))
				


def create_tf_records(anchors_df, classes):
	models_directory = os.path.sep.join([root_dir, 'models'])

	# loop over the stages (train / test)
	for stage, group in anchors_df.groupby(['stage']):
		
		# create a tensorflow record file per stage
		writer = tf.python_io.TFRecordWriter(os.path.sep.join([models_directory, stage + '.record']))
		
		# loop over the unique files in the stage
		for file_path, file_group in group.groupby(['file_path']):
			
			# create the trainings examples
			tf_example = create_tf_example(file_path, file_group, classes)
			
			# write to record
			writer.write(tf_example.SerializeToString())
		
		# close the writer
		writer.close()
		
		print('Successfully created the '+ stage + ' TFRecords')


@click.command()
@click.argument('images_directory', type=click.Path(exists=True))
def main(images_directory):

	# convert the given path to a full path
	images_directory = os.path.sep.join([root_dir, images_directory])

	# get the xml information
	anchors_df = xml_2_df(images_directory)

	# extract the classes
	classes = {name:id for id, name in enumerate(anchors_df['class'].unique())}

	# create the tensorflow records
	create_tf_records(anchors_df, classes)


if __name__ == '__main__':
	main()