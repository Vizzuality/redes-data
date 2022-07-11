import os
import io
import json
import requests
from PIL import Image

import ee
import requests
import numpy as np
import xarray_leaflet
import ipyleaflet as ipyl
from shapely.geometry import shape

from .utils import from_np_to_xr, normalize_01, normalize_m11, denormalize_01, denormalize_m11 
from . import ee_collection_specifics
from .models.CNN.super_resolution import srgan

class Predictor:
    """
    Predictions with Deep Learning models.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    model_name: string
        Name of the model
    models: Model
        List of Keras models
    """
    def __init__(self, folder_path, dataset_name, models):
        self.folder_path = folder_path
        self.dataset_name = dataset_name
        self.models = models
        with open(os.path.join(folder_path, dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)

        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

    def select_region(self, lat=39.31, lon=0.302, zoom=6):
        """
        Returns a leaflet map with the composites and allow area selection.
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
            composites.append(ee_collection_specifics.Composite(slug)(self.init_date, self.end_date))

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

    def predict(self, norm_range=[[0,1], [-1,1]]):
        """
        Predict output.
        Parameters
        ----------
        norm_range: list
            List with two values showing the normalization range.
        """
        # Normalize input image
        if norm_range[0] == [0,1]:
            self.image = normalize_01(self.image)
        elif norm_range[0] == [-1,1]:
            self.image = normalize_m11(self.image)
        else:
            raise ValueError(f'Normalization range should be [0,1] or [-1,1]')

        self.predictions = []
        for n, model in enumerate(self.models): 
            prediction = model.predict(self.image[:,:,:,:3])

            # Display predicted image on map
            # Denormalize output image
            if norm_range[1] == [0,1]:
                prediction = denormalize_01(prediction)
            elif norm_range[1] == [-1,1]:
                prediction = denormalize_m11(prediction)

            self.predictions.append(prediction)

            xda = from_np_to_xr(prediction[0,:,:,:], self.region, layer_name = f'Prediction {str(n)}')
            l = xda.leaflet.plot(self.map, rgb_dim='band', persist=True)
