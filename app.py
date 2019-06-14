# Python modules
import io
import dash
import numpy
import base64

# 3rd party modules
from PIL import Image
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Custom modules
from flowers import inference


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


def objects_2_cards(objects, nr_columns=3):
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


# =============================================================================
# Layout
# =============================================================================


title = 'Flower detector'

card_content = [
	dbc.CardHeader("<type> <species>"),
	dbc.CardImg(src='assets/images/placeholder286x180.png', top=True),
	dbc.CardBody([
		html.P("Accuracy: <accuracy>", className="card-text"),
	]),
]

navbar = dbc.NavbarSimple(children=[], brand=title, brand_href="https://www.axians.nl/data-science/", sticky="top")
body = dbc.Container([
	dbc.Row([
		dcc.Upload(
			id='input_image',
			children=html.Div(['Drag and Drop / ', html.A('Select images'), ' of bouquets']),
			multiple=False
		)
	], id='inputs'),
	dbc.Row([
		dbc.Col([html.Div(id='output_image')], md=4),
		dbc.Col([html.Div(id='output_objects')], md=8),
	])
], className="mt-4")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([navbar, body])
app.title = title


# =============================================================================
# Callbacks
# =============================================================================


@app.callback(Output('output_image', 'children'), [Input('input_image', 'contents')])
def output_image(contents):
	if contents is not None:
		return html.Div([
			html.H2("Input"),
			dbc.Card([
				dbc.CardImg(src=contents, top=True),
				# dbc.CardBody([html.H4("Card title", className="card-title"), html.P("Some quick example text to build on the card title and make up the bulk of the card's content.", className="card-text"), dbc.Button("Go somewhere", color="primary")]),
			])
		])

	# fallback (no content)
	return ''


@app.callback(Output('output_objects', 'children'), [Input('input_image', 'contents')])
def output_objects(contents):
	if contents is not None:

		image = src_2_img(contents)
		objects = inference(image)
		cards = objects_2_cards(objects)
		
		return html.Div([
			html.H2("Flowers"),
			dbc.Row(cards)
		])

	# fallback (no content)
	return ''


if __name__ == "__main__":
	app.run_server(debug=True, host='0.0.0.0', port=8050)