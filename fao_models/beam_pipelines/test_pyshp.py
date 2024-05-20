# works but i don't think we'll be able ot make centroid lat lon with this package easily
import shapefile

input_file = 'C:\\Users\\kyle\\Downloads\\ALL_centroids_completed_v1_\\ALL_centroids_completed_v1_.shp'
sf = shapefile.Reader(input_file)
print(sf.fields)
print(sf.records()[0:10])