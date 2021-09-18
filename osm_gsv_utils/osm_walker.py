'''
Created on 06 Sep 2021

@author: Tyler Saxton
'''

import json
import math
import os
import sys
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
        self.ways_by_name       = {} # Dictionary giving, for each way name, a list of ways
        self.ways_by_id         = {} # Dictionary to look up the way details by the way id
        self.way_names_by_id    = {} # Dictionary to look up the way name by the way id
        self.way_names_per_node = {} # Dictionary giving, for each node, the list of way NAMEs attached to the node
        self.way_ids_per_node   = {} # Dictionary giving, for each node, the list of way IDs attached to the node
        self.nodes              = {} # Dictionary giving node objects by node id
        
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
            for tag in tags:
                k = tag.getAttribute('k')
                if k == 'name':
                    way_name = tag.getAttribute('v').upper()

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
        
                # We also want to record intersections with unnamed "junctions" e.g. roundabouts
                if k == 'junction':
                    way_name = tag.getAttribute('v').upper()

                    # Remember the name for this way, so we don't have the parse the whole way again later
                    self.way_names_by_id[way_id] = way_name
                
                    # Add this way to the list of ways by that (generic) name
                    if not intersections_only:
                        if way_name in self.ways_by_name:
                            self.ways_by_name[way_name].append(way_id)
                        else:
                            self.ways_by_name[way_name] = [way_id]
                        
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
                 
        # Cache nodes in dict by id/ref
        if not intersections_only:
            for node in nodes_xml:
                id = node.getAttribute('id').upper()
                self.nodes[id] = node
    
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
    def bearing_from_nodes(self, prev_node, next_node):
        lat1 = float(prev_node.getAttribute('lat'))
        lon1 = float(prev_node.getAttribute('lon'))
        lat2 = float(next_node.getAttribute('lat'))
        lon2 = float(next_node.getAttribute('lon'))
    
        bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
        if bearing < 0:
            bearing = bearing + 360
        
        return bearing
        
    # Get a series of points from one point to the next, at regular intervals up to a desired offset
    def expand_offsets(self, lat1, lon1, lat2, lon2, max_offset, interval, way_id, node_id):
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
                    bearing,
                    step_i * interval * polarity,
                    way_id,
                    node_id
                ]
            
                sample_points.append(sample_point)
    
        return sample_points

    # Get every sample point for a way
    def walk_way_intersections_by_id(self, way_id, min_offset=0, max_offset=0, interval=10, debug=False):
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
        
            way_names_for_node = self.way_names_per_node[ref]
        
            if len(way_names_for_node) > 1:
                if debug:
                    print('Debug Node Intersection: {0:s} {1:d} {2:.6f}, {3:.6f}'.format(ref, len(self.way_names_for_node),
                        float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon'))))
            
                # Found an intersection!  We will output for this one
            
                # Find any negative offset samples required
                if idx > idx_first and min_offset < 0 and interval > 0:
                    prev_points = self.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx-1].getAttribute('ref')].getAttribute('lon'),
                        min_offset,
                        interval,
                        way_id,
                        ref
                    )
                
                    sample_points = sample_points + prev_points[::-1] # Reversed with slicing
            
                # Find the bearing at the node itself, and output the node itself
                if idx == idx_first:
                    bearing = self.bearing_from_nodes(
                        self.nodes[node_refs[idx  ].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )
                elif idx == idx_last:
                    bearing = self.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx  ].getAttribute('ref')]
                    )
                else:
                    bearing = self.bearing_from_nodes(
                        self.nodes[node_refs[idx-1].getAttribute('ref')],
                        self.nodes[node_refs[idx+1].getAttribute('ref')]
                    )

                sample_point = [
                    float(self.nodes[ref].getAttribute('lat')), float(self.nodes[ref].getAttribute('lon')), bearing, 0, way_id, ref
                ]
                        
                sample_points.append(sample_point)
                        
                # Find any postive offset samples required
                if idx < idx_last and max_offset > 0 and interval > 0:
                    sample_points = sample_points + self.expand_offsets(
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx  ].getAttribute('ref')].getAttribute('lon'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lat'),
                        self.nodes[node_refs[idx+1].getAttribute('ref')].getAttribute('lon'),
                        max_offset,
                        interval,
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
    def is_way_start(self, way_id):
        # Retrieve the details of the way, and find the first node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        first_node_id = node_refs[0].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
        
        # Are there any other ways that intersect with this node, with any way name?
        intersecting_ways = self.way_ids_per_node[first_node_id]

        # If there is only one way linked to the first node, then it must be this
        # way, and therefore this is the start of a way
        if len(intersecting_ways) <= 1:
            return True
    
        # How many ways with the same name are linked to the first node?
        intersecting_ways_with_this_name = 0
              
        for intersecting_way_id in intersecting_ways:
            intersecting_way_name = self.way_names_by_id[intersecting_way_id]
                        
            if intersecting_way_name == this_way_name:
                intersecting_ways_with_this_name = intersecting_ways_with_this_name + 1
            
        # If there is only one way with this name linked to the first node, then it is this one!
        if intersecting_ways_with_this_name <= 1:
            return True
    
        return False
        
    # At the end of a way, does it join to another way by the same name?
    def find_next_way(self, way_id, unused_way_ids, match_name=True):
        # Retrieve the details of the way, and find the last node ID
        way           = self.ways_by_id[way_id]
        node_refs     = way.getElementsByTagName('nd')
        last_node_id  = node_refs[-1].getAttribute('ref')
        this_way_name = self.way_names_by_id[way_id]
    
        # Are there any other ways that intersect with this node, with any way name?
        intersecting_ways = self.way_ids_per_node[last_node_id]

        # See if any of them have the same name
        for intersecting_way_id in intersecting_ways:
            if intersecting_way_id != way_id and intersecting_way_id in unused_way_ids:
                if match_name:
                    intersecting_way_name = self.way_names_by_id[intersecting_way_id]
            
                    if intersecting_way_name == this_way_name:
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
                points = self.walk_way_intersections_by_id(way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
    
            all_points = all_points + points

            return all_points
    
        # Process ways in name order, so we can walk down streets that are divided into multiple ways in an
        # order that appears sensible to a human
    
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
            unused_way_ids = way_ids.copy()
                    
            # Process the next way start
            while len(way_starts) > 0:
                # Process the next start way id
                way_id = way_starts.pop()
            
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s}'.format('START', way_id, self.way_names_by_id[way_id], len(unused_way_ids)))
            
                # Find the points we want to add to the sample for this way id, and add them
                points = self.walk_way_intersections_by_id(way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
                all_points = all_points + points
            
                # Remove this way start from the list of ways we have not processed yet
                unused_way_ids.remove(way_id)
            
                # Recursively Walk down ajoining ways in order
                next_way_id = self.find_next_way(way_id, unused_way_ids, match_name=True)
            
                while next_way_id:
                    way_id = next_way_id
                
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s}'.format('CONT', way_id, self.way_names_by_id[way_id], len(unused_way_ids)))
                      
                    # Find the points we want to add to the sample for this way id, and add them
                    points = self.walk_way_intersections_by_id(way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
                    all_points = all_points + points
            
                    # Remove this way start from the list of ways we have not processed yet
                    unused_way_ids.remove(way_id)
                    if way_id in way_starts:
                        way_starts.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, unused_way_ids, match_name=True)
                
            # Handle any unused ways that weren't a "way start" and weren't linked to one
            while len(unused_way_ids) > 0:
                # Process the next way id
                way_id = unused_way_ids.pop()
            
                if verbose:
                    print('{3:10d} {0:5s} {1:10s} {2:s}'.format('NEXT', way_id, self.way_names_by_id[way_id], len(unused_way_ids)))
                          
                # Find the points we want to add to the sample for this way id, and add them
                points = self.walk_way_intersections_by_id(way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
                all_points = all_points + points
            
                # Recursively walk down any ajoining ways in order
                next_way_id = self.find_next_way(way_id, unused_way_ids, match_name=False)
            
                while next_way_id:
                    way_id = next_way_id
                
                    if verbose:
                        print('{3:10d} {0:5s} {1:10s} {2:s}'.format('CONT', way_id, self.way_names_by_id[way_id], len(unused_way_ids)))
                          
                    # Find the points we want to add to the sample for this way id, and add them
                    points = self.walk_way_intersections_by_id(way_id, min_offset=min_offset, max_offset=max_offset, interval=interval)
            
                    all_points = all_points + points
            
                    # Remove this way start from the list of ways we have not processed yet
                    unused_way_ids.remove(way_id)
                
                    next_way_id = self.find_next_way(way_id, unused_way_ids, match_name=False)
                
        return all_points