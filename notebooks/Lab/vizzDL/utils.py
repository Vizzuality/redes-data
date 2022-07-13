import os
import ee
import json
import glob
import ffmpeg
import rioxarray
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
from shapely.geometry import shape
from google.cloud import storage
from google.oauth2 import service_account

from . import ee_collection_specifics

def polygons_to_geoStoreMultiPoligon(Polygons):
    Polygons = list(filter(None, Polygons))
    MultiPoligon = {}
    properties = ["training", "validation", "test"]
    features = []
    for n, polygons in enumerate(Polygons):
        multipoligon = []
        for polygon in polygons.get('features'):
            multipoligon.append(polygon.get('geometry').get('coordinates'))
            
        features.append({
            "type": "Feature",
            "properties": {"name": properties[n]},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates":  multipoligon
            }
        }
        ) 
        
    MultiPoligon = {
        "geojson": {
            "type": "FeatureCollection", 
            "features": features
        }
    }

    # Add bbox
    bboxs = []
    for feature in MultiPoligon.get('geojson').get('features'):
        bboxs.append(list(shape(feature.get('geometry')).bounds))
    bboxs = np.array(bboxs)
    bbox = [min(bboxs[:,0]), min(bboxs[:,1]), max(bboxs[:,2]), max(bboxs[:,3])]

    MultiPoligon['bbox'] = bbox

    return MultiPoligon

def get_geojson_string(geom):
    coords = geom.get('coordinates', None)
    if coords and not any(isinstance(i, list) for i in coords[0]):
        geom['coordinates'] = [coords]
    feat_col = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}
    return json.dumps(feat_col)

def GeoJSONs_to_FeatureCollections(geostore):
    feature_collections = []
    for n in range(len(geostore.get('geojson').get('features'))):
        # Make a list of Features
        features = []
        for i in range(len(geostore.get('geojson').get('features')[n].get('geometry').get('coordinates'))):
            features.append(
                ee.Feature(
                    ee.Geometry.Polygon(
                        geostore.get('geojson').get('features')[n].get('geometry').get('coordinates')[i]
                    )
                )
            )
            
        # Create a FeatureCollection from the list.
        feature_collections.append(ee.FeatureCollection(features))
    return feature_collections

def check_status_data(task, file_paths):
    status_list = list(map(lambda x: str(x), task.list()[:len(file_paths)])) 
    status_list = list(map(lambda x: x[x.find("(")+1:x.find(")")], status_list))
    
    return status_list

def list_record_features(glob):
    """
    Identify features in a TFRecord.
    """
    # Dict of extracted feature information
    features = {}
    # Iterate records
    glob = tf.compat.v1.io.gfile.glob(glob)
    for rec in tf.data.TFRecordDataset(glob, compression_type='GZIP'):
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features


def from_np_to_xr(array, bbox, layer_name = ''):
    """
    Transform from numpy array to geo-referenced xarray DataArray.
    Parameters
    ----------
    array: numpy array
        Numpy array with (y,x,band) dimensions.
    bbox: list
        Bounding box [min_x, min_y, max_x, max_y].
    """
    lon_coor = np.linspace(bbox[0],  bbox[2], array.shape[1])
    lat_coor = np.linspace(bbox[3],  bbox[1], array.shape[0])

    for i in range(array.shape[2]):
        xda_tmp = xr.DataArray(array[:,:,i], dims=("y", "x"), coords={"x": lon_coor, "y":lat_coor})
        if i == 0:
            xda = xda_tmp.assign_coords({"band": i})
        else:
            xda_tmp = xda_tmp.assign_coords({"band": i})
            xda = xr.concat([xda, xda_tmp], dim='band')
            
    xda.rio.write_crs(4326, inplace=True)
    xda = xda.rio.write_nodata(0)
    xda = xda.astype('uint8')
    xda.name = layer_name

    return xda

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_01(x):
    """Inverse of normalize_m11."""
    return (x * 255.0).astype(int)

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return ((x + 1) * 127.5).astype(int)

def display_lr_hr_sr(model, lr, hr):

    lr = tf.cast(lr, tf.float32).numpy()
    hr = tf.cast(hr, tf.float32).numpy()

    sr = model.predict(lr)

    fig, ax = plt.subplots(1, 3, figsize=(15,10))

    ax[0].imshow(denormalize_01(lr[0,:,:,:]))
    ax[0].set_title('LR')

    ax[1].imshow(denormalize_m11(sr[0,:,:,:]))
    ax[1].set_title('SR')

    ax[2].imshow(denormalize_m11(hr[0,:,:,:]))
    ax[2].set_title('HR')


def from_TMS_to_XYZ(tile_dir, minZ, maxZ):
    
    for z in range(minZ,maxZ+1):
        z_dir = os.path.join(tile_dir, str(z))
        all_y = [y for y in range(2**z)]
        x_list = os.listdir(z_dir)    
        for x in x_list:
            x_dir = os.path.join(z_dir, x)
            y_list = [yy.split('.')[0] for yy in os.listdir(x_dir)] 
            for y in y_list:
                old_file = os.path.join(x_dir, f"{str(y)}.png")
                new_file = os.path.join(x_dir, f"{str(all_y[-int(y)-1])}_new.png")
                os.rename(old_file, new_file)
            for filename in glob.iglob(os.path.join(x_dir, '*_new.png')):
                os.rename(filename, filename[:-8] + '.png')

def create_movie_from_pngs(input_files, output_file, output_format='apng', framerate=4):
    
    if output_format == 'apng':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate, pix_fmt='rgba')
            .output(output_file, **{'plays': 0})
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        #ffmpeg -framerate 3 -i ./data/movie/movie_%03d.png -plays 0 ./data/movie/movie.apng
        
    if output_format == 'webm':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate, pix_fmt='yuva420p')
            .output(output_file, **{'auto-alt-ref': 0})
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -framerate 4 -start_number 000 -i ./data/movie/movie_%03d.png -vf scale=490:512 -c:v libvpx -pix_fmt yuva420p -auto-alt-ref 0 -metadata:s:v:0 alpha_mode="1" ./data/movie/movie.webm
       
    if output_format == 'webm':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate, pix_fmt='yuva420p')
            .output(output_file, **{'auto-alt-ref': 0})
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -framerate 4 -start_number 000 -i ./data/movie/movie_%03d.png -vf scale=490:512 -c:v libvpx -pix_fmt yuva420p -auto-alt-ref 0 -metadata:s:v:0 alpha_mode="1" ./data/movie/movie.webm
        
    if output_format == 'webp':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate)
            .output(output_file, vcodec='libwebp')
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -framerate 4 -i ./data/movie/movie_%03d.png -c:v libwebp ./data/movie/movie.webp
        
    if output_format == 'gif':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate)
            .output(output_file, vcodec='gif')
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -f image2 -framerate 1 -i ./data/movie/movie_%03d.png  ./data/movie/movie.gif
        
    if output_format == 'mp4':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate, pix_fmt='yuv420p')
            .output(output_file, vcodec='libx264', crf=20)
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -framerate 4 -start_number 000 -i ./data/movie/movie_%03d.png -vf scale=490:512 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p movie.mp4

    if output_format == 'mov':
        out, _ = (
            ffmpeg
            .input(input_files, framerate=framerate, pix_fmt='rgba')
            .output(output_file, vcodec='png')
            .run(capture_stdout=True, overwrite_output=True)
        )
        # Corresponding command line code
        # ffmpeg -framerate 4 -start_number 000 -i ./data/movie/movie_%03d.png -vf scale=490:512 -vcodec png -pix_fmt rgba ./data/movie/movie.mov

def upload_local_directory_to_gcs(bucket_name, local_path, destination_blob_path):
    """Uploads a directory to the bucket."""
    private_key = json.loads(os.getenv("EE_PRIVATE_KEY"))
    
    credentials = service_account.Credentials.from_service_account_info(private_key)
    storage_client = storage.Client(project='project_id', credentials=credentials)
    
    bucket = storage_client.bucket(bucket_name)
    rel_paths = glob.glob(local_path + '/**', recursive=True)

    for local_file in rel_paths:
        remote_path = f'{destination_blob_path}{"/".join(local_file.split(os.sep)[len(local_path.split(os.sep))-1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            print(
                "File {} uploaded to {}.".format(
                    local_file, remote_path
                )
            )
            blob.upload_from_filename(local_file)