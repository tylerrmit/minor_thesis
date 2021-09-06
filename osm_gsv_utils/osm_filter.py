'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import geojson
from geojson import Feature, FeatureCollection, dump


class osm_filter(object):
    '''
    A class used to filteran OpenStreetMap extract down to a particular locality
    '''

    def __init__(self, filename):
        '''
        Parameters
        ----------
        filename : str
            The name of the GeoJSON file to load.
            Either a full path, or assume it's relative to the current working directory
        '''
        with open(filename) as json_file:
            self.gj = geojson.load(json_file)

        json_file.close()

    def save_locality(self, filename, filter_value, filter_key='vic_loca_2'):
        # Find the locality
        gj_index    = 0
        gj_selected = 0

        while True:
            try:
                gj_item = self.gj[gj_index]

                gj_index = gj_index + 1

                if gj_item['properties'][filter_key].upper() == filter_value.upper():
                    gj_selected = gj_item
                    break

            except Error:
                break

        if gj_selected == 0:
            raise Exception("Could not find locality")

        # Save the locality to a new file
        features = []
        features.append(gj_selected)
        feature_collection = FeatureCollection(features)

        with open(filename, 'w') as output_file:
            dump(feature_collection, output_file)

