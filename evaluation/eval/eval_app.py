import ipdb

import os
import json
import torchvision
import torch

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
# from os.path import dirname, realpath, join
# # from ..metrics.inception_score import InceptionScore
# # from ..utils.utils import printProgressBar
# from ..datasets.datamanager import AudioDataManager
# from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
#     parse_state_name, getNameAndPackage, saveScore

# # from ..networks.constant_net import FeatureTransform

# from .train_inception_model import SpectrogramInception3

# # from ..metrics.frechet_inception_distance import FrechetInceptionDistance
# import ipdb
# from torch.utils.data import DataLoader
# import numpy as np

# from audio.tools import saveAudioBatch

# import matplotlib.pyplot as plt 
# from ..utils.rainbowgram.wave_rain import wave2rain
# from ..utils.rainbowgram.rain2graph import rain2graph

PITCHES = ['A', 'B', 'C']
PATH_TO_MODELS = "/Users/javier/Developer/gans/gan_zoo_audio/output_networks"

def list_folders(path):
    print(path)
    return [f for f in os.listdir(path) if os.path.isdir(f)]


def load_model(name, scale, config, module, **kwargs):
   # # Loading the model
   #  scale = getVal(kwargs, "scale", None)
    
       # Are all parameters available ?
    # name = getVal(kwargs, "name", None)
    if name is None and not kwargs['selfNoise']:
        raise ValueError("You need to input a name")

    # module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    if name is not None:
        iteration = getVal(kwargs, "iter", None)

        checkPointDir = os.path.join(kwargs["dir"], name)
        checkpointData = getLastCheckPoint(
            checkPointDir, name, scale=scale, iter=iteration)

        if checkpointData is None:
            print(scale, iteration)
            if scale is not None or iteration is not None:
                raise FileNotFoundError("Not checkpoint found for model "
                                        + name + " at directory " + dir +
                                        " for scale " + str(scale) +
                                        " at iteration " + str(iteration))
            raise FileNotFoundError(
                "Not checkpoint found for model " + name + " at directory "
                + dir)

        modelConfig, pathModel, _ = checkpointData
        with open(modelConfig, 'rb') as file:
            configData = json.load(file)        


        modelPackage, modelName = getNameAndPackage(module)
        modelType = loadmodule(modelPackage, modelName)

        model = modelType(useGPU=False,
                          # storeAVG=True,
                          storeAVG=False,
                          **configData)

        if scale is None or iter is None:
            _, scale, iteration = parse_state_name(pathModel)

        print("Checkpoint found at scale %d, iter %d" % (scale, iteration))
        model.load(pathModel)

    elif scale is None:
        raise AttributeError("Please provide a scale to compute the noise of \
        the dataset")

    return model


def test(parser, visualisation=None):

    kwargs = vars(parser.parse_args())
    config = getVal(kwargs, "config", None)
 
    with open(config, 'rb') as file:
            config = json.load(file)
    
    model = load_model(**kwargs)
    latentDim = model.config.categoryVectorDim


    batchSize = 10
    nBatch = 5
    path_to_raw = "/ldaphome/jnistal/data/nsynth-train/audio/"
    path_out = "/ldaphome/jnistal/sandbox"
    
    path_to_raw = "/Users/javier/Developer/datasets/nsynth-train/audio"
    path_out = "/Users/javier/Developer/sandbox"

    # HACK: load a data manager for the inversion of the cqt
    loaderConfig = config["dataConfig"]
    del loaderConfig["path_to_raw"]
    del loaderConfig["path_out"]
    del loaderConfig["db_size"]
    data_manager = AudioDataManager(path_to_raw=path_to_raw,
                                    path_out=path_out,
                                    db_size=10,
                                    preprocess=False,
                                    **loaderConfig)
    data_loader = data_manager.get_loader()

    n_samples = 5

    # RANDOM GENERATION
    print("Generating audio...")
    z_noise = model.buildNoiseData(batchSize)[0]
    gen_batch = model.test(z_noise,
                           toCPU=True, getAvG=True)
    audio_out = [data_manager.postprocess(x) for x in gen_batch]
    # saveAudioBatch(audio_out, path=path_out, basename='test')
            

    # # SINGLE TIMBRE / PITCH SWEEP
    n_dim = configData['dimLatentVector']
    n_pitch = z_noise.size(1) - configData['dimLatentVector']


#################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Button('Generate', id='btn-1', n_clicks_timestamp=0),
        html.Div(id='container-button-timestamp')
    ]),

    dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in PITCHES],
                value='Life expectancy at birth, total (years)'
            ),
    dcc.Slider(
        id='year--slider',
        min=0,
        max=1,
        value=0,
        # marks={str(year): str(year) for year in df['Year'].unique()},
        step=0.1    
    ),
    # DIV MODEL UPLOAD
    dcc.Dropdown(
            id='model-folders',
            options=[{'label': i, 'value': i} for i in list_folders(PATH_TO_MODELS)],
            value='Life expectancy at birth, total (years)'
        ),
    html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                ],
                layout=go.Layout(
                    title='US Export of Plastic Scrap',
                    showlegend=True,
                    legend=go.layout.Legend(
                        x=0,
                        y=1.0
                    ),
                    margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                )
            ),
            style={'height': 300},
            id='my-graph'
        )  
    ])
])


# DRAG AND DROP
@app.callback(Output('my-graph', 'figure'),
              [Input('model-folders', 'value')])
              # [State('upload-model', 'filename'),
              #  State('upload-model', 'last_modified')])
def update_output(list_of_contents):
    print(list_of_contents)
    # if list_of_contents is not None:
    #     children = [
    #         # parse_contents(c, n, d) for c, n, d in
    #         # zip(list_of_contents, list_of_names, list_of_dates)
    #     ]
    #     return children


# # BUTTON CALLBACK
# @app.callback(
#     output=Output('my-graph', component_property='figure'),
#     # Output(component_id='my-div', component_property='children'),
#     inputs=[Input(component_id='btn-1', component_property='n_clicks')]
# )
# def update_output_div(input_value):

#     return 'You\'ve entered "{}"'.format(input_value)




if __name__ == '__main__':
    app.run_server(debug=True)

