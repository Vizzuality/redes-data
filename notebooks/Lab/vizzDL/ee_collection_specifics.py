
"""
Information on Earth Engine collections stored here (e.g. bands, collection ids, etc.)
"""

import ee
import datetime
import numpy as np

def ee_collections(collection):
    """
    Earth Engine image collection names
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': 'COPERNICUS/S2',
        'Landsat-4-Surface-Reflectance': 'LANDSAT/LT04/C01/T1_SR',
        'Landsat-5-Surface-Reflectance': 'LANDSAT/LT05/C01/T1_SR',
        'Landsat-7-Surface-Reflectance': 'LANDSAT/LE07/C01/T1_SR',
        'Landsat-8-Surface-Reflectance': 'LANDSAT/LC08/C01/T1_SR',
        'Landsat-457-Surface-Reflectance': ['LANDSAT/LT04/C01/T1_SR','LANDSAT/LT05/C01/T1_SR', 'LANDSAT/LE07/C01/T1_SR'],
        'USDA-NASS-Cropland-Data-Layers': 'USDA/NASS/CDL',
        'USGS-National-Land-Cover-Database': 'USGS/NLCD'
    }
    
    return dic[collection]

def ee_bands(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','B8A','B8','B11','B12','NDVI','NDWI'],
        'Landsat-4-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','NDVI','NDWI'],
        'Landsat-5-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','NDVI','NDWI'],
        'Landsat-7-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','NDVI','NDWI'],
        'Landsat-457-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','NDVI','NDWI'],
        'Landsat-8-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','B10','B11','NDVI','NDWI'],
        'USDA-NASS-Cropland-Data-Layers': ['landcover', 'cropland', 'land', 'water', 'urban'],
        'USGS-National-Land-Cover-Database': ['impervious']
    }
    
    return dic[collection]

def ee_bands_rgb(collection):
    """
    Earth Engine rgb band names
    """
    
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': ['B4','B3','B2'],
        'Landsat-4-Surface-Reflectance': ['B3','B2','B1'],
        'Landsat-5-Surface-Reflectance': ['B3','B2','B1'],
        'Landsat-7-Surface-Reflectance': ['B3','B2','B1'],
        'Landsat-457-Surface-Reflectance': ['B3','B2','B1'],
        'Landsat-8-Surface-Reflectance': ['B4', 'B3', 'B2'],
        'USDA-NASS-Cropland-Data-Layers': ['landcover'],
        'USGS-National-Land-Cover-Database': ['impervious']
    }
    
    return dic[collection]

def ee_bands_names(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': ['RGB','NDVI', 'NDWI'],
        'Landsat-4-Surface-Reflectance': ['RGB','NDVI', 'NDWI'],
        'Landsat-5-Surface-Reflectance': ['RGB','NDVI', 'NDWI'],
        'Landsat-7-Surface-Reflectance': ['RGB','NDVI', 'NDWI'],
        'Landsat-457-Surface-Reflectance': ['RGB','NDVI', 'NDWI'],        
        'Landsat-8-Surface-Reflectance': ['RGB','NDVI', 'NDWI'],
        'USDA-NASS-Cropland-Data-Layers': ['landcover', 'cropland', 'land', 'water', 'urban'],
        'USGS-National-Land-Cover-Database': ['impervious'],
    }
    
    return dic[collection]

def ee_scale(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': 10,
        'Landsat-4-Surface-Reflectance': 30,
        'Landsat-5-Surface-Reflectance': 30,
        'Landsat-7-Surface-Reflectance': 30,
        'Landsat-457-Surface-Reflectance': 30,        
        'Landsat-8-Surface-Reflectance': 30,
        'USDA-NASS-Cropland-Data-Layers': 30,
        'USGS-National-Land-Cover-Database': 30,
    }
    
    return dic[collection]

def vizz_params_rgb(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': {'min':0,'max':3000, 'bands':['B4','B3','B2']},
        'Landsat-4-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
        'Landsat-5-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
        'Landsat-7-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
        'Landsat-457-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
        'Landsat-8-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B4','B3','B2']},
        'USDA-NASS-Cropland-Data-Layers': {'min':0,'max':3, 'bands':['landcover']},
        'USGS-National-Land-Cover-Database': {'min': 0, 'max': 1, 'bands':['impervious']},
    }
    
    return dic[collection]

def vizz_params(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': [{'min':0,'max':3000, 'bands':['B4','B3','B2']},
            {'min':-1,'max':1, 'bands':['NDVI']},
            {'min':-1,'max':1, 'bands':['NDWI']}],
        'Landsat-4-Surface-Reflectance': [{'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDVI']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDWI']}],
        'Landsat-5-Surface-Reflectance': [{'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDVI']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDWI']}],
        'Landsat-7-Surface-Reflectance': [{'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDVI']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDWI']}],
        'Landsat-457-Surface-Reflectance': [{'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDVI']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDWI']}],        
        'Landsat-8-Surface-Reflectance': [{'min':0,'max':3000, 'gamma':1.4, 'bands':['B4','B3','B2']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDVI']},
            {'min':-1,'max':1, 'gamma':1.4, 'bands':['NDWI']}],
        'USDA-NASS-Cropland-Data-Layers': [{'min':0,'max':3, 'bands':['landcover']},
            {'min':0,'max':1, 'bands':['cropland']},
            {'min':0,'max':1, 'bands':['land']},
            {'min':0,'max':1, 'bands':['water']},
            {'min':0,'max':1, 'bands':['urban']}],
        'USGS-National-Land-Cover-Database': [{'min': 0, 'max': 1, 'bands':['impervious']}],
    }
    
    return dic[collection]

def time_steps(collection):
    """
    Composite years time steps
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': 1,
        'Landsat-457-Surface-Reflectance': 4,
        'Landsat-8-Surface-Reflectance': 1,
    }
    
    return dic[collection]


def step_range(collection):
    """
    Composite years time step ranges
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': [-1,0],
        'Landsat-457-Surface-Reflectance': [-2,2],
        'Landsat-8-Surface-Reflectance': [-1,0],
    }
    
    return dic[collection]

def date_range(instrument):
    """
    Instruments data ranges
    """
    now = datetime.datetime.now()
    dic = {
        'Landsat': {'Landsat-457-Surface-Reflectance': np.arange(1985, 2012+1), 'Landsat-8-Surface-Reflectance': np.arange(2013, now.year)},
        'Sentinel': {'Sentinel-2-Top-of-Atmosphere-Reflectance': np.arange(2016, now.year)}
    }
    
    return dic[instrument]

## ------------------------- Filter datasets ------------------------- ##
## Lansat 4, 5 and 7 Cloud Free Composite
def CloudMaskL457(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def CloudFreeCompositeL(Collection_id, startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)\
            .map(CloudMaskL457)

    ## Composite
    composite = collection.median()
    
    return composite

## Lansat 4 Cloud Free Composite
def CloudFreeCompositeL4(startDate, stopDate):
    ## Define your collections
    collection_L4 = ee.ImageCollection('LANDSAT/LT04/C01/T1_SR')

    ## Filter 
    collection_L4 = collection_L4.filterDate(startDate,stopDate).map(CloudMaskL457)

    ## Composite
    composite = collection_L4.median()
    
    return composite

## Lansat 5 Cloud Free Composite
def CloudFreeCompositeL5(startDate, stopDate):
    ## Define your collections
    collection_L5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')

    ## Filter 
    collection_L5 = collection_L5.filterDate(startDate,stopDate).map(CloudMaskL457)

    ## Composite
    composite = collectionL5.median()
    
    return composite

## Lansat 4 + 5 + 7 Cloud Free Composite
def CloudFreeCompositeL457(startDate, stopDate):
    ## Define your collections
    collection_L4 = ee.ImageCollection('LANDSAT/LT04/C01/T1_SR')
    collection_L5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    collection_L7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')

    ## Filter 
    collection_L4 = collection_L4.filterDate(startDate,stopDate).map(CloudMaskL457)
    collection_L5 = collection_L5.filterDate(startDate,stopDate).map(CloudMaskL457)
    collection_L7 = collection_L7.filterDate(startDate,stopDate).map(CloudMaskL457)
    
    ## merge collections
    collection = collection_L4.merge(collection_L5).merge(collection_L7)

    ## Composite
    composite = collection.median()
    
    return composite

## Lansat 7 Cloud Free Composite
def CloudMaskL7sr(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def CloudFreeCompositeL7(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate).map(CloudMaskL7sr)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['NDVI', 'NDWI']
    for nB, normDiff_band in enumerate([['B4','B3'], ['B4','B2']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Lansat 8 Cloud Free Composite
def CloudMaskL8sr(image):
    opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    thermalBands = ['B10', 'B11']

    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
    mask2 = image.mask().reduce('min')
    mask3 = image.select(opticalBands).gt(0).And(
            image.select(opticalBands).lt(10000)).reduce('min')
    mask = mask1.And(mask2).And(mask3)
    
    return image.updateMask(mask)

def CloudFreeCompositeL8(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate).map(CloudMaskL8sr)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['NDVI', 'NDWI']
    for nB, normDiff_band in enumerate([['B5','B4'], ['B5','B3']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Sentinel 2 Cloud Free Composite
def CloudMaskS2(image):
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
    parsed by Nick Clinton
    """
    AerosolsBands = ['B1']
    VIBands = ['B2', 'B3', 'B4']
    RedBands = ['B5', 'B6', 'B7', 'B8A']
    NIRBands = ['B8']
    SWIRBands = ['B11', 'B12']

    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # Both flags set to zero indicates clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask)

def CloudFreeCompositeS2(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('COPERNICUS/S2')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(CloudMaskS2)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['NDVI', 'NDWI']
    for nB, normDiff_band in enumerate([['B8','B4'], ['B8','B3']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Cropland Data Layers
def CroplandData(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('USDA/NASS/CDL')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    ## Change classes
    land = ['65', '131', '141', '142', '143', '152', '176', '87', '190', '195']
    water = ['83', '92', '111']
    urban = ['82', '121', '122', '123', '124']
    
    classes = []
    for n, i in enumerate([land,water,urban]):
        a = ''
        for m, j in enumerate(i):
            if m < len(i)-1:
                a = a + 'crop == '+ j + ' || '
            else: 
                a = a + 'crop == '+ j
        classes.append('('+a+') * '+str(n+1))
    classes = ' + '.join(classes)
    
    image = image.expression(classes, {'crop': image.select(['cropland'])})
    
    image =image.rename('landcover')
    
    # Split image into 1 band per class
    names = ['cropland', 'land', 'water', 'urban']
    mask = image
    for i, name in enumerate(names):
        image = ee.Image.cat([image, mask.eq(i).rename(name)])
     
    return image

## National Land Cover Database
def ImperviousData(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('USGS/NLCD')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    ## Select impervious band
    image = image.select('impervious')
    
    ## Normalize to 1
    image = image.divide(100).float()
    
    return image

## ------------------------------------------------------------------- ##

def Composite(collection):
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': CloudFreeCompositeS2,
        'Landsat-4-Surface-Reflectance': CloudFreeCompositeL,
        'Landsat-5-Surface-Reflectance': CloudFreeCompositeL,
        'Landsat-7-Surface-Reflectance': CloudFreeCompositeL7,
        'Landsat-457-Surface-Reflectance': CloudFreeCompositeL457,
        'Landsat-8-Surface-Reflectance': CloudFreeCompositeL8,
        'USDA-NASS-Cropland-Data-Layers': CroplandData,
        'USGS-National-Land-Cover-Database': ImperviousData
    }
    
    return dic[collection]

