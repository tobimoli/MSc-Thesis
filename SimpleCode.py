from osgeo import gdal

filepath = r"P:\11202428-hisea-inter\Tobias_Molenaar\01-Data\DAP\Data\SW00_OPT_MS4_1C_20180612T092821_20180612T092823_TOU_1234_a2dc.DIMA\IMG_SW00_OPT_MS4_1C_20180612T092821_20180612T092823_TOU_1234_a2dc_R1C1.TIF"

# Open the file:
raster = gdal.Open(filepath)

# Check type of the variable 'raster'
type(raster)