'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import math
import os
import sys

import pandas as pd

import xml.dom.minidom

from geographiclib.geodesic import Geodesic


class osm_walker(object):
    '''
    A class used to "walk" routes in an OSM extract and create a list of sample points and bearings

    '''

    def __init__(self, filename_main, filename_margin=None, verbose=False):
        '''
        Parameters
        ----------
        filename_main : str
            The name of the main OSM XML file to load
            Either a full path, or assume it's relative to the current working directory
        filename_margin: str
            The name of a second OSM XML file encompassing the first, with a "margin"
            around it to catch intersecting streets just outside the main area
        '''

        # Cache ways in dict by name
        self.ways_by_name            = {} # Dictionary giving, for each way name, a list of ways
        self.ways_by_id              = {} # Dictionary to look up the way details by the way id
        self.way_names_by_id         = {} # Dictionary to look up the way name by the way id
        self.way_names_per_node      = {} # Dictionary giving, for each node, the list of way NAMEs attached to the node
        self.way_ids_per_node        = {} # Dictionary giving, for each node, the list of way IDs attached to the node
        self.way_ends_per_node       = {} # Dictionary giving, for each node, the list of way IDs that start or end with the node
        self.nodes                   = {} # Dictionary giving node objects by node id
        
        self.node_coords             = {} # Dictionary giving [lat, lon] by node_id
        
        self.way_node_ids            = {} # Dictionary giving a list of node_ids (in order) for each way, by way_id
        
        self.linked_way_sections     = {} # Dictionary giving ways that have been re-connected by name, list of intersection node_ids in order
        self.linked_way_sections_all = {} # Dictionary giving ways that have been re-connected by name, list of ALL node_ids in order
        
        self.detection_hits          = {} # Dictionary giving detection hits from a detection log
        
        self.unused_way_ids          = [] # Keep track of any way IDs we have not yet linked to a way segment, to make sure we get them all
        
        self.features                = [] # Features we are building to draw on a map
        
        
        # Load the main XML file into memory
        # This assumes that we have reduced the OpenStreetMap data down to a small enough locality
        # that the in-memory approach is feasible
        doc_main = xml.dom.minidom.parse(filename_main)

        self.process_osm_xml(doc_main, intersections_only=False, verbose=verbose)
        
        if filename_margin:
            # Load a slightly bigger XML file into memory, to catch nodes that are JUST outside the
            # boundary of the locality
            doc_margin = xml.dom.minidom.parse(filename_margin)

            self.process_osm_xml(doc_margin, intersections_only=True, verbose=verbose)


    def process_osm_xml(self, doc, intersections_only=False, verbose=True):
        # Get ways and nodes from XML document
        ways_xml  = doc.getElementsByTagName('way')
        nodes_xml = doc.getElementsByTagName('node')

        for way in ways_xml:
            # Get the ID for this way
            way_id = way.getAttribute('id')
                            
            # Find the name for the way based on a 'tag' element where k='name'
            tags = way.getElementsByTagName('tag')
            found_name = False
            way_name   = 'Unnamed'
            
            for tag in tags:
                k = tag.getAttribute('k')
                
                # First preference is to use the actual name
                if k == 'name':
                    way_name = tag.getAttribute('v').upper()
                    found_name = True
                
                # Otherwise use generic "ROUNDABOUT" or "SERVICE" for unnamed segments
                elif not found_name and (k == 'junction' or k == 'highway'):
                    way_name = tag.getAttribute('v').upper()
                    
                    # Exclude unnamed FOOTWAY
                    # Otherwise we get a squiggle near the scout hall at Baden Powell Drive,
                    # where someone drew a meandering path in the grass.  GSV will give us
                    # closest image, which WILL see a bike logo (on the ROAD, from the ROAD)
                    # and the whole track looks like a bike path to nowhere
                    # way_id 289029036
                    if k == 'highway' and way_name == 'FOOTWAY':
                        way_name = None
                
                # Skip natural features like cliff, coastline, wood, beach, water, bay
                elif k == 'natural':
                    way_name = None
            
            if way_name is not None:
                # Remember the name for this way, so we don't have the parse the whole way again later
                self.way_names_by_id[way_id] = way_name
                
                # Add this way to the list of ways by that name
                if not intersections_only:
                    if way_name in self.ways_by_name:
                        self.ways_by_name[way_name].append(way_id)
                    else:
                        self.ways_by_name[way_name] = [way_id]
            
                    # Records the way by its way id
                    # We only add ways that have a name, implicitly excluding "natural" ways such as coastline
                    self.ways_by_id[way_id] = way
    
                # Record the association with this way against the node
                # We can tell that an intersection is a node associated with multiple ways
                node_refs = way.getElementsByTagName('nd')
                for node_ref in node_refs:
                    ref = node_ref.getAttribute('ref')
                    if ref in self.way_names_per_node:
                        if way_name not in self.way_names_per_node[ref]:
                            self.way_names_per_node[ref].append(way_name)
                    else:
                        self.way_names_per_node[ref] = [way_name]
                        
                    if ref in self.way_ids_per_node:
                        if way_id not in self.way_ids_per_node[ref]:
                            self.way_ids_per_node[ref].append(way_id)
                    else:
                        self.way_ids_per_node[ref] = [way_id]
                        
                    if way_id in self.way_node_ids:
                        self.way_node_ids[way_id].append(ref)
                    else:
                        self.way_node_ids[way_id] = [ref]
                
                node_ends = [node_refs[0], node_refs[-1]]
                for node_ref in node_ends:
                    ref = node_ref.getAttribute('ref')                        
                    if ref in self.way_ends_per_node:
                        if way_id not in self.way_ends_per_node[ref]:
                            self.way_ends_per_node[ref].append(way_id)
                    else:
                        self.way_ends_per_node[ref] = [way_id]                
                
                recorded = True
                 
        # Cache nodes in dict by id/ref
        if not intersections_only:
            for node in nodes_xml:
                id = node.getAttribute('id').upper()
                self.nodes[id] = node
                
                lat = float(node.getAttribute('lat'))
                lon = float(node.getAttribute('lon'))
                
                self.node_coords[id] = [lat, lon]
    
            if verbose:
                print('Way count:          %d' % ways_xml.length)
                print('Included ways:      %d' % len(self.ways_by_id.keys()))
                print('Way names:          %d' % len(self.ways_by_name.keys()))
                print('Node count:         %d' % nodes_xml.length)
        
        if verbose:
            # Count intersections
            intersection_count = 0
    
            for node in self.way_names_per_node.keys():
                if len(self.way_names_per_node[node]) > 1:
                    intersection_count = intersection_count + 1
    
            print('Intersection count: %d' % intersection_count)


    # Get the bearing from one node to the next
    @staticmethod
    def bearing_from_nodes(prev_node, next_node):
        lat1 = float(prev_node.getAttribute('lat'))
        lon1 = float(prev_node.getAttribute('lon'))
        lat2 = float(next_node.getAttribute('lat'))
        lon2 = float(next_node.getAttribute('lon'))
    
        bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
        if bearing < 0:
            bearing = bearing + 360
        
        return int(round(bearing))

        
    # Get a series of points from one point to the next, at regular intervals up to a desired offset
    @staticmethod
    def expand_offsets(lat1, lon1, lat2, lon2, max_offset, interval, way_id_start, way_id, node_id):
        sample_points = []
    
        bearing = Geodesic.WGS84.Inverse(float(lat1), float(lon1), float(lat2), float(lon2))['azi1']
    
        line = Geodesic.WGS84.InverseLine(float(lat1), float(lon1), float(lat2), float(lon2))
    
        num_steps     = int(math.ceil(abs(max_offset) / interval))
        num_steps_max = int(math.ceil(line.s13 / interval))
    
        if max_offset < 0:
            polarity = -1
        else:
            polarity = 1
        
        for step_i in range(num_steps + 1):
            if step_i > 0:
                s = min(interval * step_i, line.s13)
                g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            
                sample_point = [
                    g['lat2'],
                    g['lon2'],
                    int(round(bearing)),
                    step_i * interval * polarity,
                    way_id_start,
                    way_id,
                    node_id
                ]
            
                sample_points.append(sample_point)
    
        return sample_points


    # Get every sample point for a way
    def walk_way_intersections_by_id(self, way_id_start, way_id, min_offset=0, max_offset=0, interval=10, debug=False):
        # Initialise list of points that will be returned
        sample_points = []
    
        # Retrieve the way
        way = self.ways_by_id[way_id]
    
        # Iterate through nodes in the way
        node_refs = way.getElementsByTagName('nd')
    
        idx_first = 0
        idx_last  = len(node_refs) - 1
    
        for idx, node_ref in enumerate(node_refs):
            ref = node_ref.getAttribute('ref')
        
            if (self.is_intersection_node(ref)) or (idx == 0) or (idx == len(node_refs) - 1):
                if debug:
                    print('Debug Node Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node[ref]),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
            
                # Found an intersection!  We will output for this one
            
                # Find any negative offset samples required
                if idx > idx_first and min_offset < 0 and interval > 0:
                    prev_points = osm_walker.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lon'),
                        min_offset,
                        interval,
                        way_id_start,
                        way_id,
                        ref
                    )
                
                    sample_points = sample_points + prev_points[::-1] # Reversed with slicing
            
                # Find the bearing at the node itself, and output the node itself
                if idx == idx_first:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx  ].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )
                elif idx == idx_last:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx  ].getAttribute('ref')]
                    )
                else:
                    bearing = osm_walker.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )

                sample_point = [
                    float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon')), bearing, 0, way_id_start, way_id, ref
                ]
                        
                sample_points.append(sample_point)
                        
                # Find any postive offset samples required
                if idx < idx_last and max_offset > 0 and interval > 0:
                    sample_points = sample_points + osm_walker.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lon'),
                        max_offset,
                        interval,
                        way_id_start,
                        way_id,
                        ref
                    )
            else:
                if debug:
                    print('Debug Node NON-Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
        
        return sample_points

    # Given a way by its ID, are there any other ways with the same name that intersect
    # with its first node?
    def is_way_start(self, way_id, verbose=False):
        # Retrieve the details of the way, and find the first node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        first_node_id = node_refs[0].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
        
        if verbose:
            print('First node: ' + first_node_id)
            
        # Are there any other ways that join, end-to-end, with this node, with any way name?
        intersecting_ways = self.way_ends_per_node[first_node_id]

        # If there is only one way linked to the first node, then it must be this
        # way, and therefore this is the start of a way
        if len(intersecting_ways) <= 1:
            if verbose:
                print('No intersecting way ends for ' + first_node_id)
            return True
    
        # How many ways with the same name are linked to the first node?
        intersecting_ways_with_this_name = 0
              
        for intersecting_way_id in intersecting_ways:
            intersecting_way_name = self.way_names_by_id[intersecting_way_id]
            
            if verbose:
                print('Intersecting way end: ' + intersecting_way_name + ' (' + intersecting_way_id + ')')
                        
            if intersecting_way_name == this_way_name and intersecting_way_id in self.ways_by_id:
                intersecting_ways_with_this_name = intersecting_ways_with_this_name + 1
        
        if verbose:
            print('Intersecting way ends with same name: ' + str(intersecting_ways_with_this_name))
            
        # If there is only one way with this name linked to the first node, then it is this one!
        if intersecting_ways_with_this_name <= 1:
            return True
    
        return False
        
    # At the end of a way, does it join to another way by the same name?
    def find_next_way(self, way_id, match_name=True):
        # Retrieve the details of the way, and find the last node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        last_node_id  = node_refs[-1].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
    
        # Are there any other ways that intersect with this node, with any way name?
        intersecting_ways = self.way_ids_per_node[last_node_id]

        # See if any of them have the same name
        for intersecting_way_id in intersecting_ways:
            if intersecting_way_id != way_id and intersecting_way_id in self.unused_way_ids:
                if match_name:
                    intersecting_way_name = self.way_names_by_id[intersecting_way_id]
            
                    if intersecting_way_name == this_way_name and intersecting_way_id in self.ways_by_id:
                        return intersecting_way_id
                else:
                    return intersecting_way_id
    
        return None

  
    # Find intersection points (and offsets from intersections) for all ways
    # Walking down ajoining ways of the same name in a sensible order, where possible
    def sample_all_way_intersections(self, min_offset, max_offset, interval=10, ordered=False, verbose=False):
        all_points = []

        # If we don't want to try to walk in a natural order, do it the simple and fast way
        if not ordered:
            for way_id in self.ways_by_id.keys():
                points = self.walk_way_intersections_by_id(way_id, way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
    
            all_points = all_points + points

            return all_points
    
        # Process ways in name order, so we can walk down streets that are divided into multiple ways in an
        # order that appears sensible to a human
        
        # Link together way sections that belong to the same name and join end-to-end
        self.link_way_sections(verbose=verbose)
        
        # Iterate through each way section
        all_points = []
        
        for way_id_start in self.linked_way_sections.keys():
            section = self.linked_way_sections[way_id_start]
            
            if verbose:
                print('{0:s} {1:s} {2:4d}'.format(way_id_start, self.way_names_by_id[way_id_start], len(section)))
        
            section_points = []
            
            for way_id in section:
                if verbose:
                    print('==> {0:s}'.format(way_id))
                    
                section_points = section_points + self.walk_way_intersections_by_id(way_id_start, way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
            all_points = all_points + section_points
                
        return all_points
    
 
    def link_way_sections(self, verbose=False):
        # Process ways in name order, so we can walk down streets that are divided into multiple ways in an
        # order that appears sensible to a human
    
        # Reset linkages between ways of same name
        self.linked_way_sections     = {}
        self.linked_way_sections_all = {}
        
        # Iterate through each distinct way name (including generic ways like "junction")
        for way_name in self.ways_by_name.keys():
            if verbose:
                print('WAY: ' + way_name)
            
            # Find a list of way_ids with this name where the first node does not intersect
            # with another way of the same name
            way_starts = []
        
            # Find all the ways by this name
            way_ids = self.ways_by_name[way_name]
        
            for way_id in way_ids:
                # Check if this way appears to be a "starting way"
                if self.is_way_start(way_id):
                    way_starts.append(way_id)
                                    
            # Initialise a list to keep track of ways that we haven't walked yet
            self.unused_way_ids = way_ids.copy()
                    
            # Process the next way start
            while len(way_starts) > 0:
                # Process the next start way id
                way_id = way_starts.pop()
                
                section = [way_id]
                                        
                # Retrieve the details of the way, and find each node ID
                section_all = []
                
                way         = self.ways_by_id[way_id]
                node_refs   = way.getElementsByTagName('nd')
                
                for node_ref in node_refs:
                    ref = node_ref.getAttribute('ref')
                    if ref not in section_all:
                        section_all.append(ref)
                        
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s}'.format('START', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids)))
                 
                # Remove this way start from the list of ways we have not processed yet
                self.unused_way_ids.remove(way_id)
            
                # Recursively Walk down adjoining ways in order
                next_way_id = self.find_next_way(way_id, match_name=True)
            
                while next_way_id:
                    way_id = next_way_id
                    
                    section = section + [way_id]
                                    
                    way         = self.ways_by_id[way_id]
                    node_refs   = way.getElementsByTagName('nd')
                
                    for node_ref in node_refs:
                        ref = node_ref.getAttribute('ref')
                        if ref not in section_all:
                            section_all.append(ref)
                                           
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s}'.format('CONT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids)))
                                  
                    # Remove this way start from the list of ways we have not processed yet
                    self.unused_way_ids.remove(way_id)
                    if way_id in way_starts:
                        way_starts.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, match_name=True)
                
                self.linked_way_sections_all[way_id] = section_all                    
                self.linked_way_sections[way_id]     = section
                
                if verbose:
                    print('section:     {0:3d} {1:s}'.format(len(section), str(section)))
                    print('section_all: {0:3d} {1:s}'.format(len(section_all), str(section_all)))
                    
            # Handle any unused ways that weren't a "way start" and weren't linked to one
            while len(self.unused_way_ids) > 0:
                # Process the next way id
                way_id = self.unused_way_ids.pop()
                
                section = [way_id]
            
                # Retrieve the details of the way, and find each node ID
                section_all = []
                
                way         = self.ways_by_id[way_id]
                node_refs   = way.getElementsByTagName('nd')
                
                for node_ref in node_refs:
                    ref = node_ref.getAttribute('ref')
                    if ref not in section_all:
                        section_all.append(ref)
                                        
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s}'.format('NEXT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids)))
            
                # Recursively walk down any ajoining ways in order
                next_way_id = self.find_next_way(way_id, match_name=False)
            
                while next_way_id:
                    way_id = next_way_id
                    
                    section = section + [way_id]
                
                    way         = self.ways_by_id[way_id]
                    node_refs   = way.getElementsByTagName('nd')
                
                    for node_ref in node_refs:
                        ref = node_ref.getAttribute('ref')
                        if ref not in section_all:
                            section_all.append(ref)
                                            
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s}'.format('CONT', way_id, self.way_names_by_id[way_id], len(self.unused_way_ids)))
            
                    # Remove this way start from the list of ways we have not processed yet
                    self.unused_way_ids.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, match_name=False)
                
                self.linked_way_sections_all[way_id] = section_all              
                self.linked_way_sections[way_id]     = section
    
    
    def load_detection_log(self, filename):
        df = pd.read_csv(filename)
        
        for index, row in df.iterrows():
            # row['lat'], lon, bearing, way_id_start, way_id, node_id, offset_id, score, bbox_0, bbox_1, bbox_2, bbox_3
            key = '{0:.0f}-{1:.0f}'.format(row['way_id_start'], row['node_id'])

            if key in self.detection_hits:
                self.detection_hits[key].append([row['offset_id'], row['score']])
            else:
                self.detection_hits[key] = [[row['offset_id'], row['score']]]
                
 
    def is_intersection_node(self, node_id):
        if len(self.way_names_per_node[node_id]) > 1:
            return True
        else:
            return False
        
        
    def write_detected_geojson(self, name, detected_geojson_filename, intersection_skip_limit=1, verbose=False):
        self.features = []
        
        for way_id_start in self.linked_way_sections.keys():
            self.draw_way_segment(way_id_start, intersection_skip_limit=intersection_skip_limit, verbose=verbose)
        
        featurecollection = {
            'type':     'FeatureCollection',
            'name':     name,
            'features': self.features
        }
        
        print('Writing to: ' + detected_geojson_filename)
        with open(detected_geojson_filename, 'w') as outfile:
            json.dump(featurecollection, outfile, indent=4)
            outfile.close()
    
    
    def draw_way_segment(self, way_id_start, intersection_skip_limit=1, verbose=False):
        self.section_node_list = self.linked_way_sections_all[way_id_start]
        
        self.node_is_intersection = {}
        self.node_hit_detected    = {}
        self.node_hit_assumed     = {}
        
        # Determine whether each node is an intersection, whether a hit was detected
        for idx, node_id in enumerate(self.section_node_list):
            #if (self.is_intersection_node(node_id)) or (idx == 0) or (idx == len(self.section_node_list) - 1):
            if self.is_intersection_node(node_id):
                self.node_is_intersection[node_id] = 1
            else:
                self.node_is_intersection[node_id] = 0
                
            key = str(way_id_start) + '-' + str(node_id)
            
            if key in self.detection_hits:
                self.node_hit_detected[node_id] = 1
            else:
                self.node_hit_detected[node_id] = 0
            
            self.node_hit_assumed[node_id]  = 0
        
        # Infer between hits if there aren't too many missed intersections
        prev_hit = self.draw_find_next_hit(-1)
        next_hit = self.draw_find_next_hit(prev_hit)
        
        while (prev_hit is not None) and (next_hit is not None):
            missed_intersections = self.draw_count_intersection_miss_between(prev_hit, next_hit)
            
            if (missed_intersections is not None) and (missed_intersections <= intersection_skip_limit):
                for idx in range(prev_hit+1, next_hit):
                    self.node_hit_assumed[self.section_node_list[idx]] = 1 + missed_intersections
                    
            prev_hit = next_hit
            next_hit = self.draw_find_next_hit(next_hit)
            
        # Assume hits before first hit if there are no prior intersections (edge of map)
        # and there were continuous hits assumed to that point
        first_hit          = self.draw_find_next_hit(-1)
        second_hit         = self.draw_find_next_hit(first_hit)
        first_intersection = self.draw_find_next_intersection(-1)
        
        if (first_hit is not None) and (second_hit is not None) and (first_intersection is not None) and (first_intersection >= first_hit):
            missed_intersections = self.draw_count_intersection_miss_between(first_hit, second_hit)
            if missed_intersections <= intersection_skip_limit:
                for idx in range(0, first_hit):
                    self.node_hit_assumed[self.section_node_list[idx]] = 1
                    
        # Assume hits after last hit if there are no further intersections (edge of map)
        # and there were continuous hits assumed to that point
        last_hit          = self.draw_find_prev_hit(len(self.section_node_list))
        second_last_hit   = self.draw_find_prev_hit(last_hit)
        last_intersection = self.draw_find_prev_intersection(len(self.section_node_list))
        
        if (last_hit is not None) and (second_last_hit is not None) and (last_intersection is not None) and (last_intersection <= last_hit):
            missed_intersections = self.draw_count_intersection_miss_between(second_last_hit, last_hit)
            if missed_intersections <= intersection_skip_limit:
                for idx in range(last_hit+1, len(self.section_node_list)):
                    self.node_hit_assumed[self.section_node_list[idx]] = 1 + missed_intersections
           
        # Output the conclusion
        if verbose:
            for idx, node_id in enumerate(self.section_node_list):       
                print('{0:12s} {1:3d} => {2:d} {3:d} {4:d} {5:.6f}, {6:.6f}'.format(
                    node_id,
                    idx,
                    self.node_is_intersection[node_id],
                    self.node_hit_detected[node_id],
                    self.node_hit_assumed[node_id],
                    self.node_coords[node_id][0],
                    self.node_coords[node_id][1]
                ))
            
        # Draw geojson features
        # Gather list of list of coordinates -- each list of coordinates represents an unbroken path to draw
        self.coordinates_list_list = []
        
        coordinates_open = False
        coordinates      = []
        
        for idx in range(0, len(self.section_node_list)):
            node_id = self.section_node_list[idx]
            
            hit = self.node_hit_detected[node_id] + self.node_hit_assumed[node_id]
            
            if hit:
                coordinates_open = True
                coordinates      = coordinates + [[self.node_coords[node_id][1], self.node_coords[node_id][0]]]
            elif coordinates_open:
                if len(coordinates) > 1:
                    self.coordinates_list_list = self.coordinates_list_list + [coordinates]
                coordinates_open           = False
                coordinates                = []
                
        if coordinates_open:
            if len(coordinates) > 1:
                self.coordinates_list_list = self.coordinates_list_list + [coordinates]
            coordinates_open           = False
            coordinates                = []
                
        if len(self.coordinates_list_list) <= 0:
            return None
                
        for i in range(0, len(self.coordinates_list_list)):
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': (str(way_id_start) + '-' + str(i)),
                    'name': (str(way_id_start) + '-' + str(i) + '-' + str(len(self.coordinates_list_list[i]))),
                    'version': '1'
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': self.coordinates_list_list[i]
                }
            }
            
            self.features.append(feature)
        
    
    def draw_find_next_intersection(self, idx):        
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None
            
    def draw_find_prev_intersection(self, idx):        
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_is_intersection[self.section_node_list[idx]]:
                return idx
            idx -= 1
        return None

    def draw_find_next_hit(self, idx):        
        if idx is None:
            return None
            
        idx += 1
        while idx < len(self.section_node_list):
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx
            idx += 1
        return None
        
            
    def draw_find_prev_hit(self, idx):        
        if idx is None:
            return None
            
        idx -= 1
        while idx >= 0:
            if self.node_hit_detected[self.section_node_list[idx]]:
                return idx   
            idx -= 1
        return None
            
    def draw_count_intersection_miss_between(self, prev_hit, next_hit):       
        if prev_hit is None or next_hit is None:
            return None
            
        intersection_miss_between = 0
        
        for idx in range(prev_hit+1, next_hit):
            if (self.node_is_intersection[self.section_node_list[idx]]) and (self.node_hit_detected[self.section_node_list[idx]] == 0):
                intersection_miss_between += 1
                
        return intersection_miss_between
        
                
    
    