
from shapely.geometry import Polygon

def create_gridcell_from_lat_lon(lon, lat, lon_res, lat_res):
    """
    Create a grid cell polygon from a given latitude and longitude

    Parameters
    ----------
    lon : float
        Longitude of the grid cell
    lat : float
        Latitude of the grid cell
    lon_res : float
        Longitude resolution of the grid cell
    lat_res : float 
        Latitude resolution of the grid cell

    """    
    half_lon_res = lon_res / 2
    half_lat_res = lat_res / 2
    return Polygon([
        (lon - half_lon_res, lat - half_lat_res),
        (lon + half_lon_res, lat - half_lat_res),
        (lon + half_lon_res, lat + half_lat_res),
        (lon - half_lon_res, lat + half_lat_res)
    ])
