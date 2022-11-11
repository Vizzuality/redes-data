import os
import io
import json
import urllib
import requests
from PIL import Image

import ee
import requests
import numpy as np
import ipyleaflet as ipyl
from shapely.geometry import shape

from utils import ee_collection_specifics
from utils.util import from_np_to_xr

class Composite:
    """
    Composite class.
    ----------
    slug: string
        Dataset slug to display on the map.
    init_date: string
        Initial date of the composite.
    end_date: string
        Last date of the composite.
    """
    def __init__(self, slug="Sentinel-2-Top-of-Atmosphere-Reflectance", init_date='2019-01-01', end_date='2019-12-31'):
        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

        self.slug = slug
        self.init_date = init_date
        self.end_date = end_date

    @classmethod
    def Sentinel(cls, init_date='2019-01-01', end_date='2019-12-31'):
        return cls(slug="Sentinel-2-Top-of-Atmosphere-Reflectance",init_date=init_date, end_date=end_date)

    @classmethod
    def Landsat(cls, init_date='2019-01-01', end_date='2019-12-31'):
        return cls(slug="Landsat-8-Surface-Reflectance",init_date=init_date, end_date=end_date)

    @classmethod
    def DEM(cls):
        return cls(slug="SRTM-Digital-Elevation",init_date=None, end_date=None)

    def select_region(self, geometry=None, lat=39.31, lon=0.302, zoom=6):
        """
        Returns a leaflet map with the composites and allow area selection.
        Parameters
        ----------
        geometry : GeoJSON
            GeoJSON with a polygon.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
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

    def get_thumb_url(self, dimensions=None):
        """
        Get thumb url.
        ----------
        dimensions : int
            A number or pair of numbers in format WIDTHxHEIGHT Maximum dimensions of the thumbnail to render, in pixels. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling.
        """ 
        if self.geometry['features'] == []:
            raise ValueError(f'A rectangle has not been drawn on the map.')

        # Area of Interest
        self.region = self.geometry.get('features')[0].get('geometry').get('coordinates')
        self.polygon = ee.Geometry.Polygon(self.region)
        self.bounds = list(shape(self.geometry.get('features')[0].get('geometry')).bounds)

        # Min max values
        self.scale = ee_collection_specifics.ee_scale(self.slug)
        
        self.minMax = self.composite.reduceRegion(**{
            "reducer": ee.Reducer.minMax(),
            "geometry": self.polygon,
            "scale": self.scale,
            "maxPixels": 1e10
            })

        # convert image to an RGB visualization
        self.band = ee_collection_specifics.ee_bands_rgb(self.slug)
        if len(self.band) == 1:
            self.vis = ee_collection_specifics.vizz_params_rgb(self.slug)
            self.vis['min'] = self.minMax.getInfo()[f"{self.band[0]}_min"]
            self.vis['max'] = self.minMax.getInfo()[f"{self.band[0]}_max"]
        else:
            self.vis = ee_collection_specifics.vizz_params_rgb(self.slug)
        self.image = ee.Image(self.composite.visualize(**self.vis))
        
        if dimensions:
            self.image =  self.image.reproject(crs='EPSG:4326', scale=self.scale)
            visSave = {'dimensions': dimensions, 'format': 'png', 'crs': 'EPSG:3857', 'region':self.region} 
        else:
            visSave = {'scale': self.scale,'region':self.region, 'crs': 'EPSG:3857'} 

        self.url = self.image.getThumbURL(visSave)

        return self.url 


    def image_as_array(self, dimensions=None, alpha_channel=False):
        """
        Create Numpy array with 1 composite per year.
        ----------
        dimensions : int
            A number or pair of numbers in format WIDTHxHEIGHT Maximum dimensions of the thumbnail to render, in pixels. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling.
        alpha_channel : Boolean
            If True adds transparency
        """ 
        self.url = self.get_thumb_url(dimensions=dimensions)

        response = requests.get(self.url)
        self.array = np.array(Image.open(io.BytesIO(response.content))) 

        if len(self.array.shape) == 2:
            #De-normalize the array
            self.array = self.array*((self.vis['max']-self.vis['min'])/255)+self.vis['min']
        else:
            #Add alpha channel if needed
            if alpha_channel and self.array.shape[3] == 3:
                self.array = np.append(self.array, np.full((self.array.shape[0],self.array.shape[1], self.array.shape[2],1), 255), axis=3)
                self.array = self.array[:,:,:4]
            else:
                self.array = self.array[:,:,:3]
        
        return self.array

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

        xda = from_np_to_xr(self.array, self.bounds, projection="EPSG:3857")

        # Create GeoTIFF
        if len(self.band) == 1:
            tif_file = os.path.join(self.region_dir, f"{self.band[0]}.3857.tif")
        else:
            tif_file = os.path.join(self.region_dir, f"RGB.byte.3857.tif")

        #Save image as GeoTIFF
        xda.rio.to_raster(tif_file)

    def save_as_PNG(self, folder_path, region_name, dimensions=None):
        """
        Save image as PNG.
        Parameters
        ----------
        folder_path: string
            Path to the folder to save the figures.
        region_name: string
            Name of the folder to save the figures.
        dimensions : int
            A number or pair of numbers in format WIDTHxHEIGHT Maximum dimensions of the thumbnail to render, in pixels. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling.
        """
        self.folder_path = folder_path
        self.region_name = region_name
        self.region_dir = os.path.join(self.folder_path, self.region_name)

        # Create folder.
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.region_dir):
            os.mkdir(self.region_dir)

        # Get thumb url
        self.url = self.get_thumb_url(dimensions=dimensions)

        # Save image as PNG
        if len(self.band) == 1:
            png_file = os.path.join(self.region_dir, f"{self.band[0]}.3857.png")
        else:
            png_file = os.path.join(self.region_dir, f"RGB.byte.3857.png")

        urllib.request.urlretrieve(self.url, png_file)













