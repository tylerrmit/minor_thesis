# minor_thesis
Code used for Minor Thesis research

## Jupyter Notebooks

### Demos and previous work

GPU_test.ipynb = train a simple TensorFlow 2 model to recognise characters (MNIST)
as a demo/test to ensure GPU is being utilized in the environment

Google_Streetview_API.ipynb = Demonstrate how to download a GSV image for a
co-ordinate (lat/lon) and how to calculate a bearing from one co-ordinate to the next

PBN_Geocoding.ipynb = Use local Nominatim server to perform reverse-geocoding
of streets in PBN geojson file.  Superceded by Find_Intersections.ipynb
and eventually PBN_Distinct_Intersections.ipynb.

Find_Intersections.ipynb = Original notebook that uses an OSM extract of Victoria +
a local Nominatim server + the CSV data extracted from the PBN .geojson file
to reverse-geocode all "existing" routes in PBN, then find the intersections

Points_and_Bearings.ipynb = Demo of how to calculate the bearing from one co-ordinate
to the next, AND to expand the co-ordinate pair into a a list of co-ordinates
at 10m intervals

### PBN_Distinct_Intersections.ipynb
Load PBN geojson file + OSM extract of Victoria,
use local Nominatim serve to perform reverse-geocoding, use OSM to find the
intersections within the bounding box of the original street, find the bearings,
save as CSV for use in Gather_Training_Dataset.ipynb. 

### Gather_Training_Dataset.ipynb
Given a list of intersection co-ordinates along "EXISTING" PBN routes 
and the bearing at each co-ordinate, sample one co-ordinate at a time, allow manual
specification of metres forward/backward from the intersection along the bearing and
manual override of the bearing, then download and display 4x GSV images at 0/90/180/270
from the bearing.  Allow "hits" (where a bicycle lane logo etc. is found in an image)
to be recorded in a CSV "hits.csv" for later labelling.

### Filter_PBN_Existing.ipynb
Take the original PBN .geojson file and filter it
into both "EXISTING" vs not "EXISTING" (planned) so that we can draw them
as differenct colours in Map_GEOJSON.ipynb

### OSM_Filter.ipynb
Extract the "shape" of a "Locality" from a government-issued .geojson file,
and then save it to a new .geojson file.  Then, generate "osmium" commands to use
the resulting .geojson file to filter an OSM file down to a smaller one for
just that locality.

### OSM_to_GeoJSON.ipynb
Load an OSM file into memory/dict objects, find the ways that
have a "cycleway" tag, then save out as a .geojson file for drawing on a map

### Map_GEOJSON.ipynb
Given .geojson files that show bicycle networks as lines from
one co-ordinate (lat/lon) to the next, draw them on a map for comparison

### Parse_GeoJSON.ipynb
Read a version of OSM cycleway or road data that has been converted into a .geojson
via Parse_OSM.ipynb.  Turn it into a list of co-ordinate pairs along each way,
and then expand those co-ordinate pairs into a list of co-ordinates at 10m intervals
=> In "Mount Eliza" there are 1804 "ways" (road segments) and 75740 points if we
sample at 10m intervals.  At $7 USD per 1000 images and 4 images per point, this
would be $303 USD for the one town.

### GSV_Batch_Load_Test.ipynb
Test the use of gsv_loader.py to download GSV images (or ignore and re-use cached
images if they have been downloaded before) based on a CSV batch file


### Parse_OSM_Intersections_Ordered.ipynb

Read an OSM file for an area, and a second slightly larger OSM file with a bounding box "margin"
200m around the original area, to catch intersecting ways at the margins.  (From OSM_Filter.ipynb
and osmium.)  Find every intersection on every "way", and generate a csv batch file
with a list of points to sample on Google Street View, to be fed into the detection model.

## Python Classes

osm_filter.py = Called by OSM_Filter.ipynb to load a "locality" geojson file and then
filter it down to only one feature that matches the "vic_loca_2" property, then
save it as a smaller geosjon file representing the "shape" of the selected locality

osm_walker.py = Load OSM data and generate list of sample points

gsv_loader.py = Download/cache GSV images requested in a batch CSV file
node_id,offset_id,lat,lon,bearing

