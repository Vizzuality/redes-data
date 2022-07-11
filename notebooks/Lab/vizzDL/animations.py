import os
import io
import json
import requests
from PIL import Image

import ee
import requests
import numpy as np
import ipyleaflet as ipyl
from shapely.geometry import shape

from . import ee_collection_specifics
from .utils import from_np_to_xr, normalize_01, normalize_m11, denormalize_01, denormalize_m11 

class Animation:
    """
    Create animations.
    ----------
    """
    def __init__(self):
        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

    def select_region(self, instrument, geometry=None, lat=39.31, lon=0.302, zoom=6):
        """
        Returns a leaflet map with the composites and allow area selection.
        Parameters
        ----------
        instrument: string
            Name of a instrument (Landsat or Sentinel).
        geometry : GeoJSON
            GeoJSON with a polygon.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.instrument = instrument
        if instrument == 'Landsat':
            self.slug = 'Landsat-8-Surface-Reflectance'
        elif instrument == 'Sentinel':
            self.slug = 'Sentinel-2-Top-of-Atmosphere-Reflectance'
        else:
            raise ValueError("Instrument must be either 'Landsat' or 'Sentinel'")

        self.geometry = geometry
        self.lat = lat
        self.lon = lon
        self.zoom = zoom

        self.map = ipyl.Map(
            basemap=ipyl.basemap_to_tiles(ipyl.basemaps.OpenStreetMap.Mapnik),
            center=(self.lat, self.lon),
            zoom=self.zoom
            )

        composite = ee_collection_specifics.Composite(self.slug)('2019-01-01', '2019-12-31')

        mapid = composite.getMapId(ee_collection_specifics.vizz_params_rgb(self.slug))
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

    def create_collection(self, start_year, stop_year):
        """
        Create a GEE ImageCollection with 1 composite per year.
        ----------
        start_year : int
            First year
        stop_year : int
            Last year
        """
        self.start_year = start_year
        self.stop_year = stop_year

        # Area of Interest
        self.polygon = ee.Geometry.Polygon(self.region)

        years = np.arange(self.start_year , self.stop_year+1)
        
        dic = ee_collection_specifics.date_range(self.instrument)
        years_range = list(map(list, list(dic.values())))
        slugs = list(dic.keys())
        
        images = []
        for year in years:
            n = 0
            in_range = False
            for sub_years in years_range:
                if year in sub_years:
                    in_range = True
                    break
                n =+ 1 
                
            if not in_range:
                raise ValueError(f'Year out of range.')
                
            slug = slugs[n]
            
            self.scale = ee_collection_specifics.ee_scale(slug)
      
            # Image Visualization parameters
            vis = ee_collection_specifics.vizz_params_rgb(slug)
            
            step_range = ee_collection_specifics.step_range(slug)
            
            startDate = ee.Date(str(year+step_range[0])+'-12-31')
            stopDate  = ee.Date(str(year+step_range[1])+'-12-31')
        
            image = ee_collection_specifics.Composite(slug)(startDate, stopDate)
    
            # convert image to an RGB visualization
            images.append(image.visualize(**vis).copyProperties(image, image.propertyNames()))
            
        return images

    def video_as_array(self, start_year, stop_year, dimensions=None, alpha_channel=False):
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
        
        images = self.create_collection(start_year, stop_year)
        
        for n, image in enumerate(images):
            print(f'Image number: {str(n)}')
            image = ee.Image(image)

            if dimensions:
                image =  image.reproject(crs='EPSG:4326', scale=self.scale)
                visSave = {'dimensions': dimensions, 'format': 'png', 'crs': 'EPSG:3857', 'region':self.region} 
            else:
                visSave = {'scale': self.scale,'region':self.region} 
    
            url = image.getThumbURL(visSave)
            response = requests.get(url)
            array = np.array(Image.open(io.BytesIO(response.content))) 
            
            array = array.reshape((1,) + array.shape)
            
            #Add alpha channel if needed
            if alpha_channel and array.shape[3] == 3:
                array = np.append(array, np.full((array.shape[0],array.shape[1], array.shape[2],1), 255), axis=3)
            
            if n == 0:
                arrays = array
            else:
                arrays = np.append(arrays, array, axis=0)
        
        self.images = arrays
        return self.images

    def predict(self, model, norm_range=[[0,1], [-1,1]]):
        """
        Predict output.
        Parameters
        ----------
        models: Model
            Keras model
        norm_range: list
            List with two values showing the normalization range.
        """
        # Normalize input image
        if norm_range[0] == [0,1]:
            self.images = normalize_01(self.images)
        elif norm_range[0] == [-1,1]:
            self.images = normalize_m11(self.images)
        else:
            raise ValueError(f'Normalization range should be [0,1] or [-1,1]')

        for n in range(self.images.shape[0]):
            if n == 0:
                self.prediction = model.predict(self.images[:1,:,:,:3])
            else:
                self.prediction = np.concatenate((self.prediction, model.predict(self.images[(n-1):n,:,:,:3])), axis=0)

        # Denormalize output images
        if norm_range[1] == [0,1]:
            self.prediction = denormalize_01(self.prediction)
        elif norm_range[1] == [-1,1]:
            self.prediction = denormalize_m11(self.prediction)

        return self.prediction

    def create_tiles(self, folder_path, region_name):
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

        for n in range(self.prediction.shape[0]):
            xda = from_np_to_xr(self.prediction[n,:,:,:], self.bounds)
            xda.rio.to_raster(os.path.join(self.region_dir, f"RGB.byte.4326.{str(n)}.tif"))





