import os
import io
import json
import shutil
import requests
from PIL import Image

import ee
import requests
import numpy as np
import ipyleaflet as ipyl
from shapely.geometry import shape

from . import ee_collection_specifics
from .utils import from_np_to_xr

class Composite:
    """
    Create images.
    ----------
    """
    def __init__(self):
        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

    def select_region(self, slug="Sentinel-2-Top-of-Atmosphere-Reflectance", init_date='2019-01-01', end_date='2019-12-31', geometry=None, lat=39.31, lon=0.302, zoom=6):
        """
        Returns a leaflet map with the composites and allow area selection.
        Parameters
        ----------
        slug: string
            Dataset slug to display on the map.
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        geometry : GeoJSON
            GeoJSON with a polygon.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.slug = slug
        self.init_date = init_date
        self.end_date = end_date
        self.geometry = geometry
        self.lat = lat
        self.lon = lon
        self.zoom = zoom

        self.map = ipyl.Map(
            basemap=ipyl.basemap_to_tiles(ipyl.basemaps.OpenStreetMap.Mapnik),
            center=(self.lat, self.lon),
            zoom=self.zoom
            )

        self.composite = ee_collection_specifics.Composite(self.slug)(self.init_date, self.end_date)

        mapid = self.composite.getMapId(ee_collection_specifics.vizz_params_rgb(self.slug))
        tiles_url = self.ee_tiles.format(**mapid)

        composite_layer = ipyl.TileLayer(url=tiles_url, name=self.slug)
        self.map.add_layer(composite_layer)

        control = ipyl.LayersControl(position='topright')
        self.map.add_control(control)

        if self.geometry:
            self.geometry['features'][0]['properties'] = {'style': {'color': "#2BA4A0", 'opacity': 1, 'fillOpacity': 0}}
            geo_json = ipyl.GeoJSON(
                data=self.geometry
            )
            self.map.add_layer(geo_json)

        else:
            print('Draw a rectangle on map to select and area.')

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

            self.geometry = feature_collection

        return self.map

    def image_as_array(self, dimensions=None, alpha_channel=False):
        """
        Create Numpy array with 1 composite per year.
        ----------
        start_year : int
            First year
        stop_year : int
            Last year
        dimensions : int
            A number or pair of numbers in format WIDTHxHEIGHT Maximum dimensions of the thumbnail to render, in pixels. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling.
        alpha_channel : Boolean
            If True adds transparency
        """ 
        if self.geometry['features'] == []:
            raise ValueError(f'A rectangle has not been drawn on the map.')

        # Area of Interest
        self.region = self.geometry.get('features')[0].get('geometry').get('coordinates')
        self.polygon = ee.Geometry.Polygon(self.region)
        self.bounds = list(shape(self.geometry.get('features')[0].get('geometry')).bounds)

        # convert image to an RGB visualization
        vis = ee_collection_specifics.vizz_params_rgb(self.slug)
        image = ee.Image(self.composite.visualize(**vis))
        
        if dimensions:
            image =  image.reproject(crs='EPSG:4326', scale=self.scale)
            visSave = {'dimensions': dimensions, 'format': 'png', 'crs': 'EPSG:3857', 'region':self.region} 
        else:
            self.scale = ee_collection_specifics.ee_scale(self.slug)
            visSave = {'scale': self.scale,'region':self.region} 

        url = image.getThumbURL(visSave)
        response = requests.get(url)
        array = np.array(Image.open(io.BytesIO(response.content))) 
        
        array = array.reshape((1,) + array.shape)
        
        #Add alpha channel if needed
        if alpha_channel and array.shape[3] == 3:
            array = np.append(array, np.full((array.shape[0],array.shape[1], array.shape[2],1), 255), axis=3)
            self.image = array[:,:,:,:4]
        else:
            self.image = array[:,:,:,:3]
        
        return self.image

    def save_as_GeoTIFF(self, folder_path, region_name):
        """
        Predict output.
        Parameters
        ----------
        folder_path: string
            Path to the folder to save the tiles.
        dataset_name: string
            Name of the folder to save the tiles.
        """
        self.folder_path = folder_path
        self.region_name = region_name
        self.region_dir = os.path.join(self.folder_path, self.region_name)

        # Create folder.
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.region_dir):
            os.mkdir(self.region_dir)

        xda = from_np_to_xr(self.image[0,:,:,:], self.bounds)

        # Create GeoTIFF
        tif_file = os.path.join(self.region_dir, f"RGB.byte.4326.tif")
        xda.rio.to_raster(tif_file)













