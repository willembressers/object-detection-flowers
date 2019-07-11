# Python modules
import io
import dash
import numpy
import base64

# 3rd party modules
from PIL import Image
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Custom modules
from flowers import inference
from src.bouquet import Bouquet
from src.flower import Flower


# =============================================================================
# Functions
# ============================================================================
def src_2_img(source):
	# Split incoming stream on type and data
	content_type, data = source.split(',')

	# Decode the data
	decoded = base64.b64decode(data)

	# read the bytes array
	return Image.open(io.BytesIO(decoded))


def img_2_src(image, extension='png'):
	# Convert captured image
	byteIO = io.BytesIO()
	image.save(byteIO, format='PNG')
	buffer = byteIO.getvalue()

	# Convert to base64 encoding
	data = base64.b64encode(buffer).decode('ascii')

	# Construct a file source
	return 'data:image/{ext};base64,{data}'.format(ext=extension, data=data)



def object_2_card(class_name, image, confidence):
	# convert opencv image to html src
	source = img_2_src(image)
	
	return dbc.Col(dbc.Card([
		dbc.CardHeader(class_name),
		dbc.CardImg(src=source),
		dbc.CardBody([html.P('Confidence: {confidence}%'.format(confidence=confidence), className="card-text")]),
	], color="primary", outline=True))


def objects_2_cards(objects, nr_columns=6):
	output = []

	for i in range(0, len(objects), nr_columns):
		row = []

		for object_data in objects[i:i + nr_columns]:

			row.append(object_2_card(
				object_data['class'], 
				object_data['image'], 
				object_data['confidence']
			))
		
		output.append(dbc.Row(row, className="mb-4"))

	return output


def count_flowers(objects):		
	key = [] 
	counts = []

	if objects != None:
		key, counts = numpy.unique([object['class'] for object in objects], return_counts=True)

	return key, counts


def get_dominant_colors(image):
	Z = image.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	K = 3
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# get the colors
	unique, counts = np.unique(label, return_counts=True)
	colors = ['rgb({red}, {green}, {blue})'.format(red=int(color[2]), green=int(color[1]), blue=int(color[0])) for color in center]
	
	return counts, colors


# =============================================================================
# Layout
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Flower detector'
app.layout = html.Div([
	dbc.NavbarSimple(children=[], brand='Flower detector', brand_href="https://www.axians.nl/data-science/", sticky="top"), 
	dbc.Container([
		dbc.Row([
			dcc.Upload(
				id='input_image',
				children=html.Div(['Drag and Drop / ', html.A('Select images'), ' of bouquets']),
				multiple=False
			),
		], id='inputs'),
		dbc.Row([
			dbc.Col([html.H2("Bouquet"), html.Div(id='bouquet')], md=4),
			dbc.Col([html.H2("Metrics"), html.Div(id='dominant_colors'), html.Div(id='output_metrics')], md=8),
		]),
		dbc.Row([
			dbc.Col([html.H2("Flowers"), html.Div(id='output_objects')]),
		])
	], className="mt-4")
])


# =============================================================================
# Callbacks
# =============================================================================
@app.callback(Output('bouquet', 'children'), [Input('input_image', 'contents')])
def get_bouquet(contents):
	bouquet_card = ''

	if contents is not None:
		bouquet_card = dbc.Card([dbc.CardImg(src=contents, top=True)])
	
	return bouquet_card


@app.callback(Output('dominant_colors', 'children'), [Input('input_image', 'contents')])
def get_dominant_colors(contents):
	color_graph = ''

	if contents is not None:
		
		# load the stream data into an opencv image
		bouquet = Bouquet(contents)

		# get the dominant colors
		colors, counts, keys = bouquet.get_dominant_colors()

		# create the graph
		color_graph = dcc.Graph(
   			figure = go.Figure(
   				data = [
					go.Bar(
						x = keys,
						y = counts,
						marker = {'color':colors},
					)
				],
    			layout = go.Layout(
    				title = 'Dominant colors',
    				showlegend = False,
    				xaxis = dict(
						showgrid = False,
						showline = False,
						showticklabels = False,
						zeroline = False,
					),
					yaxis = dict(
						showgrid = False,
						showline = False,
						showticklabels = False,
						zeroline = False,
					),
    			)
    		)
    	)
	
	return color_graph


@app.callback([Output('output_objects', 'children'), Output('output_metrics', 'children')], [Input('input_image', 'contents')])
def output_objects(contents):
	output_objects = ''
	output_metrics = ''

	if contents is not None:

		image = src_2_img(contents)
		objects = inference(image)
		cards = objects_2_cards(objects)

		keys, values = count_flowers(objects)

		output_objects = html.Div([html.H2("Flowers"), dbc.Row(cards)])
		output_metrics = html.Div([html.H2("Metrics"),
			dcc.Graph(figure = {
				'data': [{"values": values, "labels": keys, "name": "flower", 'textinfo': 'value', "hoverinfo":"label+percent", "hole": .8, "type": "pie"}],
				"layout": {"title":"Flower counts"}
			})
		])

	return output_objects, output_metrics


if __name__ == "__main__":
	app.run_server(debug=True, host='0.0.0.0', port=8050)