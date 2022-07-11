import ee
import os
import json
import time
import folium
import tensorflow as tf
#from dotenv import load_dotenv
from shapely.geometry import shape
from google.oauth2 import service_account

from . import ee_collection_specifics

from .utils import polygons_to_geoStoreMultiPoligon, get_geojson_string,\
    GeoJSONs_to_FeatureCollections, check_status_data

class ee_TFRecords(object):
    """
    Create TFRecords from Google Earth Engine datasets.

    Parameters
    ----------
    folder_path: string
        Path to the folder where dataset parameters will be stored.
    dataset_name: string
        Name of the folder where dataset parameters will be stored.
    """
    def __init__(self, folder_path, dataset_name):
        self.private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
        self.credentials = service_account.Credentials(self.private_key, self.private_key['client_email'], self.private_key['token_uri'])
        self.ee_credentials = ee.ServiceAccountCredentials(email=self.private_key['client_email'], key_data=os.getenv("EE_PRIVATE_KEY"))
        self.slugs_list = ["Sentinel-2-Top-of-Atmosphere-Reflectance",
              "Landsat-7-Surface-Reflectance",
              "Landsat-8-Surface-Reflectance",
              "USDA-NASS-Cropland-Data-Layers",
              "USGS-National-Land-Cover-Database"]
        self.ee_tiles = '{tile_fetcher.url_format}'
        self.bucket = os.getenv("GCSBUCKET")
        self.project_id = os.getenv("PROJECT_ID")
        self.folder = 'Redes' # folder path to save the data
        self.colors = ['#02AEED', '#7020FF', '#F84B5A', '#FFAA36']
        self.style_functions = [lambda x: {'fillOpacity': 0.0, 'weight': 4, 'color': color} for color in self.colors]

        self.params = {}
        self.params['folder_path'] = folder_path
        self.params['dataset_name'] = dataset_name

        ee.Initialize(credentials=self.ee_credentials)

    def composite(self, slugs=["Sentinel-2-Top-of-Atmosphere-Reflectance"], init_date='2019-01-01', end_date='2019-12-31', lat=39.31, lon=0.302, zoom=6):
        """
        Returns a folium map with the composites.
        Parameters
        ----------
        slugs: list
            A list of dataset slugs to display on the map.
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.params['slugs'] = slugs
        self.params['init_date'] = init_date
        self.params['end_date'] = end_date
        self.lat = lat
        self.lon = lon
        self.zoom = zoom


        map = folium.Map(
                location=[self.lat, self.lon],
                zoom_start=self.zoom,
                tiles='OpenStreetMap',
                detect_retina=True,
                prefer_canvas=True
        )

        composites = []
        for n, slug in enumerate(slugs):
            composites.append(ee_collection_specifics.Composite(slug)(init_date, end_date))

            mapid = composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))
            tiles_url = self.ee_tiles.format(**mapid)
            folium.TileLayer(
            tiles=tiles_url,
            attr='Google Earth Engine',
            overlay=True,
            name=slug).add_to(map)

        self.composites = composites

        map.add_child(folium.LayerControl())
        return map

    def create_geostore_from_geojson(self, attributes, zoom=6):
        """Parse valid geojson into a geostore object and register it to a
        Gestore object on a server. 
        Parameters
        ----------
        attributes: list
            List of geojsons with the trainig, validation, and testing polygons.
        zoom: int
            A z-level for the map.
        """
        # Get MultiPolygon geostore object
        self.params['geostore'] = polygons_to_geoStoreMultiPoligon(attributes)

        nFeatures = len(self.params['geostore'].get('geojson').get('features'))

        nPolygons = {}
        for n in range(nFeatures):
            multipoly_type = self.params['geostore'].get('geojson').get('features')[n].get('properties').get('name')
            nPolygons[multipoly_type] = len(self.params['geostore'].get('geojson').get('features')[n].get('geometry').get('coordinates'))
    
        for multipoly_type in nPolygons.keys():
            print(f'Number of {multipoly_type} polygons:', nPolygons[multipoly_type])

        self.nPolygons = nPolygons
        self.params['nPolygons'] = self.nPolygons

        # Returns a folium map with the polygons
        features = self.params['geostore']['geojson']['features']
        if len(features) > 0:
            shapely_geometry = [shape(feature['geometry']) for feature in features]
        else:
            shapely_geometry = None
    
        self.centroid = list(shapely_geometry[0].centroid.coords)[0][::-1]
    
        bbox = self.params['geostore'].get('bbox')
        self.bounds = [bbox[2:][::-1], bbox[:2][::-1]]  

        # Returns a folium map with normalized images
        map = folium.Map(location=self.centroid, zoom_start=zoom)
        map.fit_bounds(self.bounds)

        for n, slug in enumerate(self.params['slugs']):
            # Get composite
            image = self.composites[n]

            for nBands, params in enumerate(ee_collection_specifics.vizz_params(slug)):
                mapid = image.getMapId(params)
                folium.TileLayer(
                tiles=self.ee_tiles.format(**mapid),
                attr='Google Earth Engine',
                overlay=True,
                name=slug + ': ' + ee_collection_specifics.ee_bands_names(slug)[nBands]
              ).add_to(map)

        nFeatures = len(features)
        for n in range(nFeatures):
            folium.GeoJson(data=get_geojson_string(features[n]['geometry']), style_function=self.style_functions[n],\
                 name=features[n].get('properties').get('name')).add_to(map)
        
        map.add_child(folium.LayerControl())
        return map

    def select_bands(self, input_bands, output_bands, input_rgb_bands=[], output_rgb_bands=[],\
                     new_input_bands=[], new_output_bands=[]):
        """
        Selects input and output bands.
        Parameters
        ----------
        input_bands: list
            List of input bands.
        output_bands: list
            List of output bands.
        input_rgb_bands: list
            List of new input RGB band names.
        output_rgb_bands: list
            List of new output RGB band names.
        new_input_bands: list
            List of new input band names.
        new_output_bands: list
            List of new output band names.
        """
        if ('RGB' in input_bands) and (input_bands[0] != 'RGB'):
            raise ValueError(f'RGB should be the first band of the input bands list')

        if ('RGB' in input_bands) and (input_rgb_bands == []):
            raise ValueError(f'Specify the name of each RGB band in the input_rgb_bands list')

        if ('RGB' in output_bands) and (output_bands[0] != 'RGB'):
            raise ValueError(f'RGB should be the first band of the output bands list')

        if ('RGB' in output_bands) and (output_rgb_bands == []):
            raise ValueError(f'Specify the name of each RGB band in the output_rgb_bands list')

        self.bands = [input_bands, output_bands]
        self.new_bands = [input_rgb_bands + new_input_bands, output_rgb_bands + new_output_bands]
        if input_rgb_bands:
            self.params['input_rgb_bands'] = input_rgb_bands
            self.params['output_rgb_bands'] = output_rgb_bands

    def stack_images(self, feature_collections):
        """
        Stack the 2D images (input and output images of the Neural Network) 
        to create a single image from which samples can be taken.
        """
        for n, slug in enumerate(self.params['slugs']):

        # Stack composite images
            if n == 0:
                for i, band in enumerate(self.bands[n]):
                    if band == 'RGB':
                        image_rgb = self.composites[n].visualize(**ee_collection_specifics.vizz_params_rgb(slug))
                        if i == 0:
                            in_image_stack = image_rgb
                        else:
                            in_image_stack = ee.Image.cat([in_image_stack, image_rgb])
                    else:
                        if i == 0:
                            in_image_stack = self.composites[n].select([band])
                        else:
                            in_image_stack = ee.Image.cat([in_image_stack, self.composites[n].select([band])])

                if self.new_bands[n]:
                    self.bands[n] = self.new_bands[n]
                    in_image_stack = in_image_stack.rename(self.new_bands[n])
            else:
                for i, band in enumerate(self.bands[n]):
                    if band == 'RGB':
                        image_rgb = self.composites[n].visualize(**ee_collection_specifics.vizz_params_rgb(slug))
                        if i == 0:
                            out_image_stack = image_rgb
                        else:
                            out_image_stack = ee.Image.cat([out_image_stack, image_rgb])
                    else:
                        if i == 0:
                            out_image_stack = self.composites[n].select([band])
                        else:
                            out_image_stack = ee.Image.cat([out_image_stack, self.composites[n].select([band])]) 

                if self.new_bands[n]:
                    self.bands[n] = self.new_bands[n]
                    out_image_stack = out_image_stack.rename(self.new_bands[n])
            
        image_stack = ee.Image.cat([in_image_stack, out_image_stack])
        self.image_stack = image_stack.float()

        self.params['in_bands'] = self.bands[0]
        self.params['out_bands'] = self.bands[1]

        if self.params['kernel_size'] == 1:
            self.params['base_names'] = ['training_pixels', 'validation_pixels', 'test_pixels']
            # Sample pixels
            vector = image_stack.sample(region = feature_collections[0], scale = self.params['scale'],\
                                        numPixels=self.params['sample_size'], tileScale=4, seed=999)

            # Add random column
            vector = vector.randomColumn(seed=999)

            # Partition the sample approximately 60%, 20%, 20%.
            self.training_dataset = vector.filter(ee.Filter.lt('random', 0.6))
            self.validation_dataset = vector.filter(ee.Filter.And(ee.Filter.gte('random', 0.6),\
                                                            ee.Filter.lt('random', 0.8)))
            self.test_dataset = vector.filter(ee.Filter.gte('random', 0.8))

            # Training and validation size
            self.training_size = self.training_dataset.size().getInfo()
            self.validation_size = self.validation_dataset.size().getInfo()
            self.test_size = self.test_dataset.size().getInfo()

        if self.params['kernel_size'] > 1:
            self.params['base_names'] = ['training_patches', 'validation_patches', 'test_patches']
            # Convert the image into an array image in which each pixel stores (kernel_size x kernel_size) patches of pixels for each band.
            list = ee.List.repeat(1, self.params['kernel_size'])
            lists = ee.List.repeat(list, self.params['kernel_size'])
            kernel = ee.Kernel.fixed(self.params['kernel_size'], self.params['kernel_size'], lists)

            self.arrays = self.image_stack.neighborhoodToArray(kernel)

            # Training and validation size
            nFeatures = len(self.params['geostore'].get('geojson').get('features'))
            nPolygons = {}
            for n in range(nFeatures):
                multipoly_type = self.params['geostore'].get('geojson').get('features')[n].get('properties').get('name')
                nPolygons[multipoly_type] = len(self.params['geostore'].get('geojson').get('features')[n].get('geometry').get('coordinates'))

            self.training_size = nPolygons['training']*self.params['sample_size']
            self.validation_size = nPolygons['validation']*self.params['sample_size']
            self.test_size = nPolygons['test']*self.params['sample_size']

        # Add values to params
        self.params['training_size'] = self.training_size
        self.params['validation_size'] = self.validation_size
        self.params['test_size'] = self.test_size

    def start_TFRecords_task(self, feature_collections, feature_lists):
        """
        Create TFRecord's exportation task
        """
        # These numbers determined experimentally.
        nShards  = int(self.params['sample_size']/20) # Number of shards in each polygon.

        if self.params['kernel_size'] == 1:
            # Export all the training validation and test data.   
            self.file_paths = []
            for n, dataset in enumerate([self.training_dataset, self.validation_dataset, self.test_dataset]):

                self.file_paths.append(self.bucket+ '/' + self.folder + '/' + self.params['dataset_name'] + '/' + self.params['base_names'][n])

                # Create the tasks.
                task = ee.batch.Export.table.toCloudStorage(
                  collection = dataset,
                  description = 'Export '+self.params['base_names'][n],
                  fileNamePrefix = self.folder + '/' + self.params['dataset_name'] + '/' + self.params['base_names'][n],
                  bucket = self.bucket,
                  fileFormat = 'TFRecord',
                  selectors = self.bands[0] + self.bands[1])

                task.start()

        if self.params['kernel_size'] > 1:
             # Export all the training validation and test data. (in many pieces), with one task per geometry.     
            self.file_paths = []
            for i, feature in enumerate(feature_collections):
                for g in range(feature.size().getInfo()):
                    geomSample = ee.FeatureCollection([])
                    for j in range(nShards):
                        sample = self.arrays.sample(
                            region = ee.Feature(feature_lists[i].get(g)).geometry(), 
                            scale = self.params['scale'], 
                            numPixels = self.params['sample_size'] / nShards, # Size of the shard.
                            seed = j,
                            tileScale = 8
                        )
                        geomSample = geomSample.merge(sample)

                    desc = self.params['base_names'][i] + '_g' + str(g)

                    self.file_paths.append(self.bucket+ '/' + self.folder + '/' + self.params['dataset_name'] + '/' + desc)

                    task = ee.batch.Export.table.toCloudStorage(
                        collection = geomSample,
                        description = desc, 
                        bucket = self.bucket, 
                        fileNamePrefix = self.folder + '/' + self.params['dataset_name'] + '/' + desc,
                        fileFormat = 'TFRecord',
                        selectors = self.bands[0] + self.bands[1]
                    )
                    task.start()

        return task

    def export_TFRecords(self, scale, sample_size, kernel_size):
        """
        Export TFRecords to GCS.
        Parameters
        ----------
        scale: float
            Scale of the images.
        sample_size: int
            Number of samples to extract from each polygon.
        kernel_size: int
            An integer specifying the height and width of the 2D images.
        scaling_factor: int
            Scaling Factor for Super-Resolution.
        """
        self.params['scale'] = scale
        self.params['sample_size'] = sample_size
        self.params['kernel_size'] = kernel_size
        self.params['data_dir'] = 'gs://' + self.bucket + '/' + self.folder + '/' + self.params['dataset_name']

        # Convert the GeoJSON to feature collections
        feature_collections = GeoJSONs_to_FeatureCollections(self.params['geostore'])
        
        # Convert the feature collections to lists for iteration.
        feature_lists = list(map(lambda x: x.toList(x.size()), feature_collections))

        # Stack the 2D images to create a single image from which samples can be taken
        self.stack_images(feature_collections)

        # Start the task
        task = self.start_TFRecords_task(feature_collections, feature_lists)

        # Save task status
        print('Exporting TFRecords to GCS:')
        status_list = check_status_data(task, self.file_paths)
        while not status_list == ['COMPLETED'] * len(self.file_paths):
            status_list = check_status_data(task, self.file_paths)
            #Save temporal status in params.json
            tmp_status = json.dumps(dict(zip(self.file_paths, status_list)))
            self.params['data_status'] = tmp_status
            print('Temporal status: ', tmp_status)

            time.sleep(60)

        # Save final status in params.json
        self.params['data_status'] = "COMPLETED"  
        print('Final status: COMPLETED')

        # Save the parameters in a json file.
        if not os.path.isdir(os.path.join(self.params['folder_path'], self.params['dataset_name'])):
            os.mkdir(os.path.join(self.params['folder_path'], self.params['dataset_name']))
        with open(os.path.join(self.params['folder_path'], self.params['dataset_name'], "dataset_params.json"), 'w') as f:
            json.dump(self.params, f)


class read_TFRecords():
    """
    Read TFRecords.
    Parameters
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    normalize_rgb: boolean
        Boolean to normalize RGB bands.
    norm_range: list
        List with two values showing the normalization range.
    batch_size: int
        A number of samples processed before the model is updated. 
        The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
    shuffle_size: int
        Number of samples to be shuffled.
    scaling_factor: int
        Scaling Factor for Super-Resolution.
    """
    def __init__(self, folder_path, dataset_name, normalize_rgb=True, norm_range = [[0,1], [-1,1]], batch_size=32, shuffle_size=2000, scaling_factor=None):
        with open(os.path.join(folder_path, dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)

        self.params['normalize_rgb'] = normalize_rgb
        self.params['norm_range'] = norm_range
        self.params['batch_size'] = batch_size
        self.params['shuffle_size'] = shuffle_size
        self.params['scaling_factor'] = scaling_factor

    def parse_function(self, proto):
        """The parsing function.
        Read a serialized example into the structure defined by features_dict.
        Args:
          example_proto: a serialized Example.
        Returns: 
          A dictionary of tensors, keyed by feature name.
        """
        
        # Define your tfrecord 
        features = self.params.get('in_bands') + self.params.get('out_bands')
        
        # Specify the size and shape of patches expected by the model.
        kernel_shape = [self.params.get('kernel_size'), self.params.get('kernel_size'), 1]
        columns = [
          tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in features
        ]
        features_dict = dict(zip(features, columns))
        
        # Load one example
        parsed_features = tf.io.parse_single_example(proto, features_dict)

        # Separate the output images from the input images
        if not self.params.get('normalize_rgb'):
            image = tf.concat([parsed_features[i] for i in self.params.get('in_bands')], axis=2)
            label = tf.concat([parsed_features[i] for i in self.params.get('out_bands')], axis=2)
        # Normalize RGB bands
        else:
            if self.params.get('input_rgb_bands') != []:
                image_rgb = tf.concat([parsed_features[i] for i in self.params.get('input_rgb_bands')], axis=2)
                if self.params['norm_range'][0] == [0,1]:
                    image_rgb = tf.divide(image_rgb, 255.0)
                elif self.params['norm_range'][0] == [-1,1]:
                    image_rgb = tf.divide(image_rgb - 127.5, 127.5)
                else:
                    raise ValueError(f'Normalization range should be [0,1] or [-1,1]')

                in_bands = self.params.get('in_bands').copy()
                [in_bands.remove(item) for item in self.params.get('input_rgb_bands')]
                if in_bands:
                    image = tf.concat([parsed_features[i] for i in in_bands], axis=2)
                    image = tf.concat([image_rgb, image], axis=2)
                else:
                    image = image_rgb

            if self.params.get('output_rgb_bands') != []:
                label_rgb = tf.concat([parsed_features[i] for i in self.params.get('output_rgb_bands')], axis=2)
                if self.params['norm_range'][1] == [0,1]:
                    label_rgb = tf.divide(label_rgb, 255.0)
                elif self.params['norm_range'][1] == [-1,1]:
                    label_rgb = tf.divide(label_rgb - 127.5, 127.5)
                else:
                    raise ValueError(f'Normalization range should be [0,1] or [-1,1]')

                out_bands = self.params.get('out_bands').copy()
                [out_bands.remove(item) for item in self.params.get('output_rgb_bands')]
                if out_bands:
                    label = tf.concat([parsed_features[i] for i in out_bands], axis=2)
                    label = tf.concat([label_rgb, label], axis=2)
                else:
                    label = label_rgb

        # Resize input images
        if self.params['scaling_factor'] is not None:
            kernel_size = int(self.params.get('kernel_size')/self.params['scaling_factor'])
            image = tf.image.resize(image, [kernel_size, kernel_size],  method='nearest')
            
        return image, label
    
    def get_dataset(self, glob):
        """Get the preprocessed training dataset
        Returns: 
        A tf.data.Dataset of training data.
        """
        glob = tf.compat.v1.io.gfile.glob(glob)
        
        dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
        dataset = dataset.map(self.parse_function, num_parallel_calls=5)
        
        return dataset

    def get_training_dataset(self):
        """Get the preprocessed training dataset
        Returns: 
        A tf.data.Dataset of training data.
        """
        glob = self.params.get('data_dir') + '/' + self.params.get('base_names')[0] + '*'
        dataset = self.get_dataset(glob)
        dataset = dataset.shuffle(self.params.get('shuffle_size')).batch(self.params.get('batch_size')).repeat()
        return dataset
    
    def get_validation_dataset(self):
        """Get the preprocessed validation dataset
        Returns: 
          A tf.data.Dataset of validation data.
        """
        glob = self.params.get('data_dir') + '/' + self.params.get('base_names')[1] + '*'
        dataset = self.get_dataset(glob)
        dataset = dataset.batch(1).repeat()
        return dataset
    
    def get_test_dataset(self):
        """Get the preprocessed validation dataset
        Returns: 
          A tf.data.Dataset of validation data.
        """
        glob = self.params.get('data_dir') + '/' + self.params.get('base_names')[2] + '*'
        dataset = self.get_dataset(glob)
        dataset = dataset.batch(1).repeat()
        return dataset

