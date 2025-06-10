

import os

# Import required libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from ClimProjTools.utils import create_gridcell_from_lat_lon

import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import from_origin

import multiprocessing as mp



def extract_livneh_lusu_data(basins_list, 
                             years_start_end, 
                             livneh_lusu_dir, 
                             output_dir,
                             livneh_lusu_temp_file_template='livneh_lusu_2020_temp_and_wind.2021-05-02.{}.nc',
                             livneh_lusu_prcp_file_template='livneh_unsplit_precip.2021-05-02.{}.nc',
                             overwrite_existing=False,
                             tmp_work_dir='../temporary_working_dir/',
                             plot_grid_cells=True
                             ):
    """
    Extracts Livneh-Lusu data for a given basin and saves it to a CSV file.

    To extract the Livneh-Lusu data, you need to have the netCDF files available in the specified directory.
    The data can be downloaded from the Livneh-Lusu dataset from: https://cirrus.ucsd.edu/~pierce/nonsplit_precip/

    
    Parameters:
    - basins_list: GeoDataFrame containing the basin for which to extract data.
    - years_start_end: Tuple containing the start and end years for the extraction (e.g., (1950, 2018)).
    - livneh_lusu_dir: Directory containing the Livneh-Lusu netCDF files.
    - output_dir: Directory to save the extracted data.
    - livneh_lusu_temp_file_template: Template for the Livneh-Lusu Temperature netCDF file names.
    - livneh_lusu_prcp_file_template: Template for the Livneh-Lusu precipitation netCDF file names.
    - overwrite_existing: If True, overwrite existing output files.
    - tmp_work_dir: Temporary working directory for intermediate files.
    - plot_grid_cells: If True, plot the grid cells that fall within each basin. By default, the figures
    are saved in the `output_dir/figures` directory.
    """
    
    # Create the list of years to extract data for
    years = np.arange(years_start_end[0], years_start_end[1] + 1)
    
    # Open one of the Livneh-Lusu netCDF files to get grid
    nc = Dataset(
        livneh_lusu_dir + livneh_lusu_temp_file_template.format(years[0]), 'r')

    # Get the latitude and longitude variables from the Livneh-luse netCDF file
    latitudes = nc.variables['lat'][:]
    longitudes = nc.variables['lon'][:]
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Create a geodataframe for the Livneh-lusu grid cells
    grid_cells = gpd.GeoDataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten()}, geometry=None)
    lat_res = np.diff(latitudes)[0] 
    lon_res = np.diff(longitudes)[0]
    grid_cells['geometry'] = \
        [create_gridcell_from_lat_lon(lon, lat, lon_res, lat_res) \
        for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    grid_cells = grid_cells.set_geometry('geometry')
    grid_cells.crs = 'EPSG:4326'  # Set the CRS to WGS84

    # Convert the `grid_cells` GeoDataFrame to a raster format
    grid_cells_raster = rasterize(
        grid_cells.geometry,
        out_shape=(len(latitudes), len(longitudes)),
        transform=from_origin(lon_grid.min(), lat_grid.max(), lon_res, lat_res),
        fill=0,
        dtype=np.float32
    )

    # Save into a raster file. This raster will be used to mask the grid cells that fall within each basin.
    if not os.path.exists(tmp_work_dir):
        os.makedirs(tmp_work_dir)

    with rasterio.open(
        os.path.join(tmp_work_dir, 'Livneh_Lusu_grid_cells_EPSG4326.tif'),
        'w',
        driver='GTiff',
        height=grid_cells_raster.shape[0],
        width=grid_cells_raster.shape[1],
        count=1,
        dtype=grid_cells_raster.dtype,
        crs='EPSG:4326',
        transform=from_origin(lon_grid.min()-lon_res/2, lat_grid.max()+lon_res/2, lon_res, lat_res)
    ) as dst:
        dst.write(grid_cells_raster, 1)



    # Loop over the basins
    for index, basin in basins_list.iterrows():
        
        # Get the basin ID and ensure it is zero-padded to 8 digits
        basin_id = str(basin['hru_id']).zfill(8)  # Ensure the basin ID is zero-padded to 8 digits

        # Check if the an output file already exists for the basin
        if not overwrite_existing:
            if os.path.exists(os.path.join(output_dir, f'livneh_lusu_basin_{basin_id}.csv')):
                print(f'Skipping basin {basin_id} as output file already exists.')
                continue
        
        # Create a mask for the basin
        with rasterio.open(os.path.join(tmp_work_dir, 'Livneh_Lusu_grid_cells_EPSG4326.tif')) as src:
            
            # Get the basin geometry in GeoJSON format
            shapes = [feature.__geo_interface__ for feature in [basin.geometry]]
            
            # Mask the grid cells raster with the basin geometry.
            # `out_image` will contain the masked grid cells, and `out_transform` 
            #  will contain the transform for the masked raster.
            out_image, out_transform = mask(src, shapes)#, crop=True)
            
        # Identify the grid cells from lon_grid and lat_grid that fall within the basin 
        # (i.e. , where the mask is not zero)
        basin_mask = out_image[0] > 0  # Assuming the first band contains the mask
        
        # Get the indices of the axes of the `out_image` array
        rows, cols = np.indices((out_image[0].shape[0], out_image[0].shape[1]))
        # xs and ys are the coordinates of the grid cells in the masked raster
        xs, ys = rasterio.transform.xy(out_transform, rows, cols, offset='center')
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Get the longitude and latitude of the grid cells that fall within the basin
        lon_basin_masked = xs.flatten()[basin_mask.flatten()]
        lat_basin_masked = ys.flatten()[basin_mask.flatten()]
            
        # Plot the basin geometry and the grid cells that fall within the basin
        if plot_grid_cells:
            output_dir_figures = os.path.join(output_dir, 'figures')
            if not os.path.exists(output_dir_figures):
                os.makedirs(output_dir_figures)

            fig, ax = plt.subplots(1,1, figsize=(10, 10))
            plt.scatter(lon_basin_masked, lat_basin_masked, c='red', s=3, label='Grid cells within basin')
            gpd.GeoSeries(basin.geometry).plot(color='blue', alpha=0.5, label='Basin boundary', ax=ax)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Grid cells within basin {basin_id} ({basin.AREA/10**6:.1f} km² - {len(lon_basin_masked)} grid cells)')
            plt.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir_figures, f'livneh_lusu_basin_{basin_id}_grid_cells.png'))
            plt.close()
            #plt.show()

        # Get the indices of the grid cells that fall within the basin
        x_cell_indices = np.searchsorted(longitudes, lon_basin_masked)
        y_cell_indices = np.searchsorted(latitudes, lat_basin_masked)

        
        # Extract the Livneh-Lusu data for the grid cells that fall within the basin
        
        # Create a dataframe to store the data for the basin
        df_basin = pd.DataFrame(columns=['tmax_C', 'tmin_C', 'wind_m/s', 'prcp_mm'], 
                                index=pd.date_range(start='1950-01-01', end='2018-12-31', freq='D'))

        # Loop over the years and extract the data for the grid cells
        for year in years:
            
            # Open the netCDF files for the year
            nc_temp = Dataset(
                livneh_lusu_dir + livneh_lusu_temp_file_template.format(year), 'r')
            
            # Open the netCDF file for precipitation for the year
            nc_precip = Dataset(
                livneh_lusu_dir + livneh_lusu_prcp_file_template.format(year), 'r')
            
            # Extract the time index
            days_since_temp = nc_temp.variables['time'].units.split(' ')[-1]
            timeindex_temp = pd.to_datetime(days_since_temp) \
                + pd.to_timedelta(nc_temp.variables['time'][:], unit='D')
            
            days_since_precip = nc_precip.variables['Time'].units.split(' ')[-1]
            timeindex_precip = pd.to_datetime(days_since_precip) \
                + pd.to_timedelta(nc_precip.variables['Time'][:], unit='D')
            
            # Check if the time indices match
            if not np.array_equal(timeindex_temp, timeindex_precip):
                raise ValueError(f'Time indices do not match for year {year} in basin {basin_id}')
            
            # Extract the temperature and wind data for the grid cells that fall within the basin and 
            # calculate the basin average
            tmax = nc_temp.variables['Tmax'][:, y_cell_indices, x_cell_indices].mean(axis=(1, 2))
            tmin = nc_temp.variables['Tmin'][:, y_cell_indices, x_cell_indices].mean(axis=(1, 2))
            wind = nc_temp.variables['Wind'][:, y_cell_indices, x_cell_indices].mean(axis=(1, 2))
            # Extract the precipitation data for the grid cells that fall within the basin
            prcp = nc_precip.variables['PRCP'][:, y_cell_indices, x_cell_indices].mean(axis=(1, 2))


            # Create a DataFrame for the data for the year
            df_year = pd.DataFrame({
                'tmax_C': tmax.flatten(),
                'tmin_C': tmin.flatten(),
                'wind_m/s': wind.flatten(),
                'prcp_mm': prcp.flatten()
            }, index=timeindex_temp)

            # Add the data for the year to the basin dataframe
            df_basin = df_basin.add(df_year, fill_value=0)

            # Close the netCDF files
            nc_temp.close()
            nc_precip.close()

        # Save the basin data to a CSV file
        output_file = os.path.join(output_dir, f'livneh_lusu_basin_{basin_id}.csv')
        df_basin.to_csv(output_file, index_label='date')

if __name__ == '__main__':

    # Split the basin shapefile into chunks for parallel processing
    num_processes = 2

    # Load the shapefile of the 671 basins
    basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
    # Read the shapefile using geopandas
    basin_shapefile = gpd.read_file(basin_shapefile_file)

    basin_chunks = np.array_split(basin_shapefile, num_processes)

    # Directory where the Livneh-Lusu data is stored (user needs to download the netCDF files))
    livneh_lusu_dir = '../data/Livneh_Lusu/' 
    # Directory where the output files will be saved 
    output_dir = '../data/Livneh_Lusu_extracted_data/' 

    tmp_work_dir = '../temporary_working_dir/'

    params = [
        (
            basin_chunk,  # your specific basin (or list of basins) for this job
            (1950, 2018),  # the years range
            livneh_lusu_dir, # the directory where the Livneh-Lusu data is stored
            output_dir, # the directory where the output files will be saved
            'livneh_lusu_2020_temp_and_wind.2021-05-02.{}.nc', # Livneh-Lusu temperature and wind file template
            'livneh_unsplit_precip.2021-05-02.{}.nc', # Livneh-Lusu precipitation file template
            False,  # Overwrite flag
            tmp_work_dir, # Temporary working directory
            True    # Plot grid cells flag
        )
        for basin_chunk in basin_chunks
    ]
    

    # extract the Livneh-Lusu data for each chunk in parallel
    mp.Pool(num_processes).starmap(extract_livneh_lusu_data, params)