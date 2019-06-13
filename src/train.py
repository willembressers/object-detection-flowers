import os
import click
import pandas
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

def create_tf_records(anchors_df):
	models_directory = os.path.sep.join([root_dir, 'models'])

	# loop over the stages (train / test)
	for stage, group in anchors_df.groupby(['stage']):
		
		# create a tensorflow record file per stage
		writer = tf.python_io.TFRecordWriter(os.path.sep.join([models_directory, stage + '.record']))
		
		# loop over the unique files in the stage
		for file_path, file_group in group.groupby(['file_path']):
			
			# create the trainings examples
			tf_example = create_tf_example(file_path, file_group)
			
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
	create_tf_records(anchors_df)

	print(anchors_df.head())


if __name__ == '__main__':
	main()