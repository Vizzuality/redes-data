import os
import io
import json
import math
import shutil
import urllib
import requests
from PIL import Image

import ee
import ffmpeg
import requests
import argparse
import gdal2tiles
import numpy as np
from apng import APNG
import ipyleaflet as ipyl
from shapely.geometry import shape

from utils import ee_collection_specifics
from models.CNN.super_resolution import srgan
from utils.util import from_np_to_xr, from_TMS_to_XYZ, create_movie_from_pngs,\
    upload_local_directory_to_gcs, normalize_01, normalize_m11, denormalize_01, denormalize_m11 
    

class Animation:
    """
    Create animations.
    ----------
    """
    def __init__(self):
        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.bucket_name = os.getenv("GCSBUCKET")
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.ee_tiles = '{tile_fetcher.url_format}'
        ee.Initialize(credentials=self.ee_credentials)

    def select_region(self, instrument, geometry=None, lat=39.31, lon=0.302, zoom=6, show_map=True):
        """
        Returns a leaflet map with the composites and allow area selection.
        Parameters
        ----------
        instrument: string
            Name of a instrument (Landsat or Sentinel).
        geometry : GeoJSON file.
            GeoJSON with a polygon.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        show_map: bool
            Display the composites on a leaflet map.
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

        if self.geometry:
            # Read GeoJSONs
            with open(self.geometry, 'r') as j:
                self.geometry = json.loads(j.read())

        self.map = ipyl.Map(
            basemap=ipyl.basemap_to_tiles(ipyl.basemaps.OpenStreetMap.Mapnik),
            center=(self.lat, self.lon),
            zoom=self.zoom
            )

        composite = ee_collection_specifics.Composite(self.slug)('2019-01-01', '2019-12-31')

        if show_map:
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
                
            self.slug = slugs[n]
            
            self.scale = ee_collection_specifics.ee_scale(self.slug)
      
            # Image Visualization parameters
            self.vis = ee_collection_specifics.vizz_params_rgb(self.slug)
            
            step_range = ee_collection_specifics.step_range(self.slug)
            
            startDate = ee.Date(str(year+step_range[0])+'-12-31')
            stopDate  = ee.Date(str(year+step_range[1])+'-12-31')
        
            image = ee_collection_specifics.Composite(self.slug)(startDate, stopDate)
    
            # convert image to an RGB visualization
            images.append(image.visualize(**self.vis).copyProperties(image, image.propertyNames()))
            
        return images

    def save_frames_as_PGNs(self, folder_path, region_name, start_year, stop_year, dimensions=None):
        """
        Save frames as PGNs.
        ----------
        folder_path: string
            Path to the folder to save the figures.
        region_name: string
            Name of the folder to save the figures.
        start_year : int
            First year
        stop_year : int
            Last year
        dimensions : int
            A number or pair of numbers in format WIDTHxHEIGHT Maximum dimensions of the thumbnail to render, in pixels. If only one number is passed, it is used as the maximum, and the other dimension is computed by proportional scaling.
        alpha_channel : Boolean
            If True adds transparency
        """ 
        self.folder_path = folder_path
        self.region_name = region_name
        self.region_dir = os.path.join(self.folder_path, self.region_name)

        # Create folder.
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.region_dir):
            os.mkdir(self.region_dir)

        if self.geometry['features'] == []:
            raise ValueError(f'A rectangle has not been drawn on the map.')

        # Area of Interest
        self.region = self.geometry.get('features')[0].get('geometry').get('coordinates')
        self.polygon = ee.Geometry.Polygon(self.region)
        self.bounds = list(shape(self.geometry.get('features')[0].get('geometry')).bounds)
        
        images = self.create_collection(start_year, stop_year)
        
        years = np.arange(start_year, stop_year+1)
        for n, image in enumerate(images):
            print(f'Image number: {str(n)}')
            image = ee.Image(image)

            if dimensions:
                image =  image.reproject(crs='EPSG:4326', scale=self.scale)
                visSave = {'dimensions': dimensions, 'format': 'png', 'crs': 'EPSG:3857', 'region':self.region} 
            else:
                visSave = {'scale': self.scale,'region':self.region, 'crs': 'EPSG:3857'} 
    
            self.url = image.getThumbURL(visSave)

            png_file = os.path.join(self.region_dir, f"RGB.byte.3857.{str(years[n])}.png")

            urllib.request.urlretrieve(self.url, png_file)


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
        self.start_year = start_year
        self.stop_year = stop_year

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
                visSave = {'scale': self.scale,'region':self.region, 'crs': 'EPSG:3857'} 
    
            url = image.getThumbURL(visSave)
            response = requests.get(url)
            array = np.array(Image.open(io.BytesIO(response.content))) 
            
            array = array.reshape((1,) + array.shape)
            
            #Add alpha channel if needed
            if alpha_channel and array.shape[3] == 3:
                array = np.append(array, np.full((array.shape[0],array.shape[1], array.shape[2],1), 255), axis=3)
                if n == 0:
                    arrays = array[:,:,:,:4]
                else:
                    arrays = np.append(arrays, array[:,:,:,:4], axis=0)
            else:
                if n == 0:
                    arrays = array[:,:,:,:3]
                else:
                    arrays = np.append(arrays, array[:,:,:,:3], axis=0)
        
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

        # Denormalize input/output images
        if norm_range[0] == [0,1]:
            self.images = denormalize_01(self.images)
        elif norm_range[0] == [-1,1]:
            self.images = denormalize_m11(self.images)
        if norm_range[1] == [0,1]:
            self.prediction = denormalize_01(self.prediction)
        elif norm_range[1] == [-1,1]:
            self.prediction = denormalize_m11(self.prediction)

        return self.prediction

    def create_animated_tiles(self, folder_path, region_name, minZ, maxZ, save_GeoTIFF=False):
        """
        Predict output.
        Parameters
        ----------
        folder_path: string
            Path to the folder to save the tiles.
        region_name: string
            Name of the folder to save the tiles.
        minZ: int
            Min zoom level.
        maxZ: int
            Max zoom level.
        save_GeoTIFF: Boolean
            Save GeoTIFFs.
        """
        self.folder_path = folder_path
        self.region_name = region_name
        self.minZ = minZ
        self.maxZ = maxZ
        self.region_dir = os.path.join(self.folder_path, self.region_name)

        # Create folder.
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.region_dir):
            os.mkdir(self.region_dir)

        animations = {self.instrument: self.images}
        if hasattr(self, 'prediction'):
            animations['Prediction'] = self.prediction

        for name, animation in animations.items():
            print(name)
            # Create tiles per frame
            print('Create tiles per frame:')
            years = np.arange(self.start_year, self.stop_year+1)
            for n in range(animation.shape[0]):
                print(f'  year #{str(years[n])}')
                xda = from_np_to_xr(animation[n,:,:,:3], self.bounds, projection="EPSG:3857")

                # Create GeoTIFF
                tif_file = os.path.join(self.region_dir, f"RGB.byte.3857.{str(years[n])}.tif")
                xda.rio.to_raster(tif_file)

                # Create Tiles
                tile_dir = os.path.join(self.region_dir, str(n))

                options = {'zoom': f'{str(minZ)}-{str(maxZ)}',
                'nb_processes': 48,
                'tile_size': 256,
                'srs':'EPSG:3857'}

                gdal2tiles.generate_tiles(tif_file, tile_dir, **options)

                from_TMS_to_XYZ(tile_dir, minZ, maxZ)

                # Remove GeoTIFF
                if not save_GeoTIFF:
                    os.remove(tif_file)

                # Merge different frame tiles    
                if n == 0:
                    # Create folder.
                    if not os.path.isdir(os.path.join(self.region_dir, 'APNGs')):
                        os.mkdir(os.path.join(self.region_dir, 'APNGs'))
                    if not os.path.isdir(os.path.join(self.region_dir, 'APNGs', name)):
                        os.mkdir(os.path.join(self.region_dir, 'APNGs', name))
                    else:
                        shutil.rmtree(os.path.join(self.region_dir, 'APNGs', name))
                        os.mkdir(os.path.join(self.region_dir, 'APNGs', name))

                z_dirs = [d for d in os.listdir(tile_dir) if os.path.isdir(os.path.join(tile_dir, d))]
                for z_dir in z_dirs:
                    if n == 0: os.mkdir(os.path.join(self.region_dir, 'APNGs', name, z_dir))
                    for x_dir in os.listdir(os.path.join(tile_dir, z_dir)):
                        if n == 0: os.mkdir(os.path.join(self.region_dir, 'APNGs', name, z_dir, x_dir))

                        source_dir = os.path.join(tile_dir, z_dir, x_dir)
                        target_dir = os.path.join(self.region_dir, 'APNGs', name, z_dir, x_dir)
                            
                        file_names = os.listdir(source_dir)
                            
                        for file_name in file_names:
                            shutil.move(os.path.join(source_dir, file_name), target_dir)
                            number = '{:03d}'.format(n)
                            os.rename(os.path.join(target_dir, file_name), os.path.join(target_dir, file_name[:-4] + f'_{number}.png'))

                shutil.rmtree(tile_dir)

            # Create APNGs
            print('Creating APNGs')
            tile_dir = os.path.join(self.region_dir, 'APNGs', name)
            for z_dir in os.listdir(tile_dir):
                for x_dir in os.listdir(os.path.join(tile_dir, z_dir)):
                    file_names = os.listdir(os.path.join(tile_dir, z_dir, x_dir))

                    tiles = list(map(lambda x: x.split('_')[0], file_names))
                    tiles = list(set(tiles))
                    for tile in tiles:
                        png_files = list(filter(lambda x: x.split('_')[0] == tile, file_names))
                        png_files = sorted(png_files, key=lambda x: float(x.split('.')[0]))
                        png_files = [os.path.join(tile_dir, z_dir, x_dir, i) for i in png_files]

                        APNG.from_files(png_files, delay=1).save(png_files[0][:-8]+'.png')
                        #create_movie_from_pngs(png_files[0].split('_')[-2]+'_'+'%03d.png', png_files[0].split('_')[-2]+'.png', 'apng')

                        # Remove PNGs
                        [os.remove(file) for file in png_files]

            # Upload `APNG` files to GCS
            print('Uploading APNG files to GCS')
            upload_local_directory_to_gcs(self.bucket_name, tile_dir, f'Redes/Tiles/{self.region_name}/APNGs/')

            # Display tiles on map
            if name == self.instrument:
                self.map = ipyl.Map(
                    basemap=ipyl.basemap_to_tiles(ipyl.basemaps.OpenStreetMap.Mapnik),
                    center=(self.lat, self.lon),
                    zoom=self.zoom
                    )

            self.map.add_layer(ipyl.TileLayer(url=f'https://storage.googleapis.com/geo-ai/Redes/Tiles/{self.region_name}/APNGs/{name}'+'/{z}/{x}/{y}.png', name=name))

        control = ipyl.LayersControl(position='topright')
        self.map.add_control(control)
        self.map.add_control(ipyl.FullScreenControl())
        
        return self.map

    def create_movie_from_array(self, folder_path, region_name, output_format='mp4', framerate=None):
        """
        Predict output.
        Parameters
        ----------
        folder_path: string
            Path to the folder to save the tiles.
        region_name: string
            Name of the folder to save the tiles.
        output_format: string
            Video format (mp4, mov or webm).
        framerate: int
            Number of frames per second.
        """
        self.folder_path = folder_path
        self.region_name = region_name
        self.output_format = output_format
        self.framerate = framerate
        self.region_dir = os.path.join(folder_path, region_name)

        # Create folder.
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.region_dir):
            os.mkdir(self.region_dir)

        animations = {self.instrument: self.images}
        if hasattr(self, 'prediction'):
            animations['Prediction'] = self.prediction

        for name, animation in animations.items():
            self.output_file = os.path.join(self.region_dir, region_name+'_'+name+'.'+output_format)

            if not isinstance(animation, np.ndarray):
                animation = np.asarray(animation)
                
            n,height,width,channels = animation.shape
            
            # Height and width should be divisible by 2
            if height % 2 == 1: height = math.floor(height / 2.) * 2
            if width % 2 == 1: width = math.floor(width / 2.) * 2
                
            animation = animation[:,:height,:width,:]
            
            if self.framerate == None:
                self.framerate = n
                
            if self.output_format == 'mp4':
                process = (
                    ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                        .output(self.output_file, pix_fmt='yuv420p', vcodec='libx264', r=self.framerate)
                        .overwrite_output()
                        .run_async(pipe_stdin=True, overwrite_output=True)
                )
                
            if self.output_format == 'mov':
                process = (
                    ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
                        .output(self.output_file, vcodec='png', r=self.framerate)
                        .overwrite_output()
                        .run_async(pipe_stdin=True, overwrite_output=True)
                )
                
            if self.output_format == 'webm':   
                process = (
                    ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
                        .output(self.output_file, **{'auto-alt-ref': 0, 'qmin': 0, 'qmax': 50, 'crf': 5, 'b:v': '1M'}, r=self.framerate)
                        .overwrite_output()
                        .run_async(pipe_stdin=True, overwrite_output=True)
                )

            if self.output_format == 'apng':
                process = (
                    ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
                        .output(self.output_file, **{'plays': 0})
                        .overwrite_output()
                        .run_async(pipe_stdin=True, overwrite_output=True)
                )
                # Corresponding command line code
                #ffmpeg -framerate 3 -i ./data/movie/movie_%03d.png -plays 0 ./data/movie/movie.apng
            
            if self.output_format == 'mp4': animation = animation[:,:,:,:3]
        
            for frame in animation:
                process.stdin.write(frame.astype(np.uint8).tobytes())
            process.stdin.close()
            process.wait() 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create animated-tiles')
    parser.add_argument('-i','--instrument', help="Name of a instrument (Landsat or Sentinel).",\
         default='Landsat')
    parser.add_argument('-g','--geometry', help="GeoJSON file with a polygon",\
         default="../../datasets/raw/Menongue.geojson")
    parser.add_argument('-f','--start_year', help="First year", default=1988)
    parser.add_argument('-l','--stop_year', help="Last year", default=2021)
    parser.add_argument('-m','--model', help='Keras model.',  default='srgan_generator')
    parser.add_argument('-nr','--norm_range', help='List with two values showing the normalization range.',\
            nargs='+', default=[[0,1],[-1,1]])
    parser.add_argument('-fp','--folder_path', help='Path to the folder to save the tiles.',\
            default='../../datasets/processed/Tiles/')
    parser.add_argument('-rn','--region_name', help='Name of the folder to save the tiles.',\
            default='Menongue')
    parser.add_argument('-iz','--min_z', help='Min zoom level.',\
            default=10)
    parser.add_argument('-az','--max_z', help='Min zoom level.',\
            default=14)
    parsed = vars(parser.parse_args())


    print("Creating animation object.")
    animation = Animation()
    print("Selecting region.")
    animation.select_region(instrument=parsed['instrument'], geometry=parsed['geometry'], show_map=False)
    print("Creating animation as a Numpy array.")
    video = animation.video_as_array(start_year=int(parsed['start_year']), stop_year=int(parsed['stop_year']))
    print("Predict.")
    if parsed['model'] == 'srgan_generator':
        # Location of model weights
        weights_dir = '../../datasets/processed/Models/L8_S2_SR_x3/srgan_generator_L8_to_S2_x3'
        weights_file = lambda filename: os.path.join(weights_dir, filename)

        pre_generator = srgan.Generator(input_shape=(None, None, 3), scale=3).generator()
        pre_generator.load_weights(weights_file('model_weights.h5'))

        animation.predict(model = pre_generator, norm_range=parsed['norm_range'])
        print("Creating animated-tiles.")
        #animation.create_animated_tiles(folder_path = parsed['folder_path'], region_name = parsed['region_name'], minZ = parsed['min_z'], maxZ=parsed['max_z'])













