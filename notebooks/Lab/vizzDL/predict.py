import os
import io 
import json
from unicodedata import name
import requests
from PIL import Image

import ee
import requests
import numpy as np
import xarray_leaflet
import ipyleaflet as ipyl
from shapely.geometry import shape

from .train import Trainer
from . import ee_collection_specifics
from .utils import from_np_to_xr


class Predictor(object):
    """
    Predictions with Deep Learning models.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    model_name: string
        Name of the model
    """
    def __init__(self, folder_path, dataset_name, model_name):
        with open(os.path.join(folder_path, dataset_name, model_name, "training_params.json"), 'r') as f:
            self.params = json.load(f)

        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.slugs_list = ["Sentinel-2-Top-of-Atmosphere-Reflectance",
              "Landsat-7-Surface-Reflectance",
              "Landsat-8-Surface-Reflectance",
              "USDA-NASS-Cropland-Data-Layers",
              "USGS-National-Land-Cover-Database"]
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

    def select_region(self, slugs=["Sentinel-2-Top-of-Atmosphere-Reflectance"], init_date='2019-01-01', end_date='2019-12-31', lat=39.31, lon=0.302, zoom=6):
        """
        Returns a folium map with the composites and allow area selection.
        Parameters
        ----------
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.lat = lat
        self.lon = lon
        self.zoom = zoom
        self.slugs = self.params['slugs']
        self.init_date = self.params['init_date']
        self.end_date = self.params['end_date']

        self.map = ipyl.Map(
            basemap=ipyl.basemap_to_tiles(ipyl.basemaps.OpenStreetMap.Mapnik),
            center=(self.lat, self.lon),
            zoom=self.zoom
            )

        composites = []
        for n, slug in enumerate(self.slugs):
            composites.append(ee_collection_specifics.Composite(slug)(init_date, end_date))

            mapid = composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))
            tiles_url = self.ee_tiles.format(**mapid)

            composite_layer = ipyl.TileLayer(url=tiles_url, name=slug)
            self.map.add_layer(composite_layer)

        self.composites = composites

        control = ipyl.LayersControl(position='topright')
        self.map.add_control(control)

        draw_control = ipyl.DrawControl()

        draw_control.rectangle = {
            "shapeOptions": {
                "color": "#2BA4A0",
                "fillOpacity": 0,
                "opacity": 1
            }
        }

        feature_collection = {
            'type': 'FeatureCollection',
            'features': []
        }

        def handle_draw(self, action, geo_json):
            """Do something with the GeoJSON when it's drawn on the map"""    
            feature_collection['features'].append(geo_json)

        draw_control.on_draw(handle_draw)
        self.map.add_control(draw_control)

        self.feature = feature_collection

        print('Draw a rectangle on map to select and area.')

        return self.map

    def create_input_image(self):
        """
        Select region on map and create input image.
        Parameters
        ----------
        """
        if self.feature['features'] == []:
            raise ValueError(f'A rectangle has not been drawn on the map.')

        self.geo = self.feature['features'][0]['geometry']
        self.polygon = shape(self.geo)

        self.region = list(self.polygon.bounds)

        visSave = ee_collection_specifics.vizz_params_rgb(self.slugs[0])
        scale = ee_collection_specifics.ee_scale(self.slugs[0])
        url = self.composites[0].getThumbURL({**visSave,**{'scale': scale}, **{'region':self.region}})

        response = requests.get(url)
        self.image = np.array(Image.open(io.BytesIO(response.content))) 
        self.image = self.image.reshape((1,) + self.image.shape) 

        # Display input image on map
        xda = from_np_to_xr(self.image[0,:,:,:], self.region, layer_name = 'Input image')
        l = xda.leaflet.plot(self.map, rgb_dim='band', persist=True)

    def predict(self):
        """
        Predict output.
        Parameters
        ----------
        """
        Train = Trainer(folder_path = self.params['folder_path'], dataset_name = self.params['dataset_name'])
        Train.create_model(model_type=self.params['model_type'], model_output=self.params['model_output'], model_architecture=self.params['model_architecture'], scaling_factor=self.params['scaling_factor'])

        model_dir = os.path.join(self.params['folder_path'], self.params['dataset_name'], self.params['model_name'], 'model_weights.h5') 
        model = Train.keras_model
        model.load_weights(model_dir)

        self.prediction = model.predict(self.image/255.)

        # Display predicted image on map
        xda = from_np_to_xr((self.prediction[0,:,:,:]*255.0).astype(int), self.region, layer_name = 'Prediction')
        l = xda.leaflet.plot(self.map, rgb_dim='band', persist=True)