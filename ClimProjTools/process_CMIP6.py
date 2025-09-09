

import glob, os
import numpy as np
import pandas as pd
import xarray as xr
import zipfile
from itertools import product
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from ClimProjTools.utils import create_gridcell_from_lat_lon
import shutil
from pathlib import Path
from scipy import stats
from statsmodels.distributions.copula.api import (CopulaDistribution, GaussianCopula)

def extract_basin_average_from_CMIP6_data(variables, experiments, basin_shapefile, epsg, 
    gcm_directory, output_directory, create_plot_gcm_grid_cells=True, lon_to_west=-360, 
    write_weights=True):

    
    """
    Extracts the basin average of a variable from CMIP6 data for a given experiment and basin shapefile.

    Parameters
    ----------
    variables : str
        The variable for which the basin average is to be extracted (currently, only 'pr', 'tas' are
        supported).

    experiments : str
        The experiment for which the data is to be extracted 
        (e.g., 'historical', 'ssp1_1_9', 'ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp4_3_4', 
               'ssp4_6_0', 'ssp5_8_5',  'ssp5_3_4os', 'ssp5_8_5').
        Note: The projections for the selected experiment must have been downloaded (e.g., using the
        'download_CNIP6.from_ClimateDataStore' function).

    basin_shapefile : str
        The path to the shapefile of the basin for which the data is to be extracted.
        Coordinates must be in WGS84 (EPSG:4326).

    epsg : int
        The EPSG code of the projection in which the data is to be projected when calculating the basin
        area. This projection is dependant on the region of the basin. For example, for basins in the
        United States, the EPSG code is 2163.

    gcm_directory : str
        The path to the directory where the CMIP6 data is stored. The data must be stored in the following
        structure: 'gcm_directory'/variable/experiment/gcm_name.zip.
        variable: the variable for which the data is stored (e.g., 'pr', 'tas').
        experiment: the experiment for which the data is stored (e.g., 'historical', 'ssp1_1_9', 'ssp1_2_6',
        'ssp2_4_5', 'ssp3_7_0', 'ssp4_3_4', 'ssp4_6_0', 'ssp5_8_5', 'ssp5_3_4os', 'ssp5_8_5').
        This structure is the same as the one used by the 'download_CMIP6.from_ClimateDataStore' function.
        Currently, all GCMs for the chosen variable will be extracted (the user cannot pick specific GCMs).

    output_directory : str
        The path to the directory where the output data is to be stored. If this directory does not exist, it
        will be created.

    create_plot_gcm_grid_cells : bool, optional
        If True, a plot of the grid cells and the basin is created for each GCM. Default is True.

    lon_to_west : int, optional
        The value to subtract from the longitude values to convert them to the western hemisphere. Default
        is -360 (i.e., the default is to convert from the eastern to the western hemisphere).

    write_weights : bool, optional
        If True, the weights of the grid cells that are inside the basin will be written to a excel file.
        Default is True.
        
    """

    # Check if one of the variable is supported
    if any(variable not in ['prcp', 'tas'] for variable in variables):
        raise ValueError("Variable not supported. Currently, only 'pr' and 'tas' are supported.")
    
    # Check if the gcm_directory exists
    if not os.path.exists(gcm_directory):
        raise ValueError(f'The directory {gcm_directory} does not exist.')
    
    # Check if the output_directory exists. If not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)    
        
    # Read the shapefile of the basin for which we want to extract the data
    shp = gpd.read_file(basin_shapefile)
    # Check if the shapefile has multiple polygons. If so, merge them into one
    if len(shp) > 1:
        basin = gpd.GeoDataFrame({'geometry':[unary_union(shp.geometry)]})
    else:    
        basin = shp

    # Set the CRS of the basin shapefile
    basin.crs = 'EPSG:4326'
    basin = basin.to_crs(epsg=epsg)

    # Find the available GCMs for each variables
    for var, exp in product(variables, experiments):

        # check if data is available for the variable and experiment
        if not os.path.exists(f'{gcm_directory}/{var}/{exp}'):
            print(f'No data available for {var} and {exp}.')
            continue

        # Grab the path of the .zip files and extract the name of the GCMs
        gcm_paths = glob.glob(f'{gcm_directory}/{var}/{exp}/*.zip')
        gcm_names = ['_'.join(g.split('_')[:-1]).split('\\')[1] for g in gcm_paths]

        # Create a dataframe to store the data
        df = pd.DataFrame(columns=['date', *gcm_names])
        df = df.set_index('date')

        if write_weights:
            weights_list = []

        # Extract the GCMs data
        for gcm_path, gcm_name in zip(gcm_paths, gcm_names):

            with zipfile.ZipFile(gcm_path, 'r') as z:
                # Unzip the archive into a temporary folder
                z.extractall(f'temporary/')
                ncfile = [i for i in z.namelist() if '.nc' in i][0]

            
            with xr.open_dataset(f'temporary/{ncfile}') as dataset:

                # Check if the coordinate' names are 'latitude' and 'longitude'. 
                # If they are, rename them to 'lat' and 'lon'
                if 'latitude' in dataset.coords:
                    dataset = dataset.rename({'latitude': 'lat'})
                if 'longitude' in dataset.coords:
                    dataset = dataset.rename({'longitude': 'lon'})

                # Extract the time and the variable values
                time = dataset.time.values.astype('datetime64[M]')

                # Calculate the fraction of the grid cell that is inside the basin

                # First, convert the dataset lat/lon coordinates into a meshgrid
                lon_grid, lat_grid = np.meshgrid(dataset.lon.values+lon_to_west, dataset.lat.values)
                # Create a GeoDataFrame for the grid cells
                grid_cells = gpd.GeoDataFrame({'lon': lon_grid.ravel(), 'lat': lat_grid.ravel(), 'geometry': None})
                lat_steps = np.diff(dataset.lat.values)
                lon_steps = np.diff(dataset.lon.values)
                if len(np.unique(lat_steps)) > 1 or len(np.unique(lon_steps)) > 1:
                    # The grid is not regular' -- we take the average of the steps. No perfect, but it will do
                    # especially since the difference is small given the resolution of the grid and the size of the basin
                    # Sometimes, the difference in 'steps' looks like a rounding error.
                    lon_res = np.mean(lon_steps)
                    lat_res = np.mean(lat_steps)
                else:
                    lon_res = lon_steps[0]
                    lat_res = lat_steps[0]
                
                grid_cells['geometry'] = \
                    [create_gridcell_from_lat_lon(lon, lat, lon_res, lat_res) \
                    for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
                
                        
                grid_cells.crs = 'EPSG:4326'
                grid_cells = grid_cells.to_crs(epsg=epsg)

                # Upon request, plot the grid cells and the basin
                if create_plot_gcm_grid_cells:
                    if not os.path.exists(f'{output_directory}/figures/GCM_grid/'):
                        os.makedirs(f'{output_directory}/figures/GCM_grid/')
                    fig, ax = plt.subplots()
                    grid_cells.to_crs(epsg=4326).boundary.plot(ax=plt.gca(), color='black')
                    basin.to_crs(epsg=4326).boundary.plot(ax=plt.gca(), color='red')
                    ax.set_xlabel('Longitude', fontsize=14)
                    ax.set_ylabel('Latitude', fontsize=14)
                    plt.title(f'{gcm_name} grid cells and basin', fontsize=16)
                    plt.tight_layout()
                    fig.savefig(
                        f'{output_directory}/figures/GCM_grid/{gcm_name}_{var}_{exp}_grid_cells_and_basin.png')
                    #plt.show()
                    plt.close()
                    

                
                # Calculate the weight to apply to each grid cell when calculating the weighted average of 
                # the variable values inside the basin
                weights = pd.DataFrame(columns=['lat', 'lon', 'weight'])
                for k, grid in enumerate(grid_cells.geometry):

                    grid = gpd.GeoDataFrame({'geometry': [grid]}, crs=epsg)
                    
                    intersections = gpd.overlay(grid, basin, how='intersection')

                    # If the intersection is empty, the fraction is 0
                    if intersections.empty:
                        weight = 0
                    # If the intersection is not empty, calculate the fraction of the grid cell that is inside the basin
                    else:
                        weight = intersections.area[0] / basin.area[0]

                    # Create a dataframe to store the weight and coordinates of the grid cell
                    weight_gridcell = pd.DataFrame({'lat': [lat_grid.ravel()[k]],
                                                    'lon': [lon_grid.ravel()[k]], 
                                                    'weight': [weight]})
                    
                    # Fill the weights dataframe that contains the weights and coordinates of all grid cells
                    if weights.empty:
                        weights = weight_gridcell
                    else:
                        weights = pd.concat([weights, weight_gridcell], ignore_index=True)

                # Add an excel sheet where the weights are stored
                if write_weights:
                    weights_list.append(
                        {'sheet_name': gcm_name, 
                            'data': {'lat': weights.lat.tolist(), 
                                    'lon': weights.lon.tolist(), 
                                    'weight': weights.weight.tolist()}}
                    )
                
                # Reshape the weights array to match the grid cells
                weights = weights.pivot(index='lat', columns='lon', values='weight')
                        
                # Calculate the weighted average of the variable values
                if var == 'tas':
                    # Reshape weights to match the dimensions of the variable values (i.e., time, lat, lon)
                    reshaped_weights = np.broadcast_to(weights.values, dataset.tas.values.shape)
                    # Calculate the weighted average of the temperature values (deg Celcius)
                    var_values = np.average(dataset.tas.values, weights=reshaped_weights, axis=(1, 2)) - 273.15
                elif var == 'prcp':
                    # Reshape weights to match the dimensions of the variable values (i.e., time, lat, lon)
                    reshaped_weights = np.broadcast_to(weights.values, dataset.pr.values.shape)
                    # Calculate the weighted average of the precipitation values (mm/day)
                    # Native units are kg m-2 s-1, so we multiply by 86400 to convert to mm/day
                    var_values = np.average(dataset.pr.values, weights=reshaped_weights, axis=(1, 2)) * 86400
                
            
            # Remove the temporary folder
            os.remove(f'temporary/{ncfile}')


            df_gcm = pd.DataFrame({'date': time, gcm_name: var_values})
            df_gcm = df_gcm.set_index('date')
            #if exp != 'historical':
            #    df_gcm.reindex(pd.date_range(start_date_ssp, end_date_ssp, freq='MS'))
            df[gcm_name] = df_gcm.values.ravel()
        
        df.index = pd.to_datetime(df_gcm.index)
        df.to_csv(f'{output_directory}/{var}_{exp}.csv')

        # Write each GCM's weights to an excel file
        if write_weights:
            with pd.ExcelWriter(f'{output_directory}/weights_{var}_{exp}.xlsx') as weights_excel:
                for gcm_name in gcm_names:
                    for weights_data in weights_list:
                        if weights_data['sheet_name'] == gcm_name:
                            df = pd.DataFrame(weights_data['data'])
                            df.to_excel(weights_excel, sheet_name=gcm_name, index=False)

        shutil.rmtree('temporary')


def calculate_delta_change(path_historical_file, path_future_file, variables, future_experiment,
                           reference_period, future_periods, output_directory):
    
    """
    Calculate the delta change of a variable between a reference period and future periods for a given GCM."

    Parameters
    ----------
    path_historical_file : str
        The path to the historical data file. If the 'path_historical_file' is None, the deltas are
        expected to be calculated between two periods of the future data.

    path_future_file : str
        The path to the future data file. This path is required and cannot be None.

    variables : list
        A list of variables for which the deltas are to be calculated. Currently, only 'tas' and 'prcp'
        are supported.

    future_experiment : str
        The name of the future experiment for which the deltas are to be calculated. Time periods for the
        experiment must be available (typically generated from the 'extract_basin_average_from_CMIP6_data'
        function). Currently, the function supports only one experiment at a time.

    reference_period : tuple of two integers (start_year, end_year)
        A tuple of two integers representing the start and end years of the reference period.
        Currently, the reference period either ends in 2014 or start in 2015. Years prior 2014 are 
        from the 'historical' experiment, while years after 2014 are from a 'future' experiment
        (e.g., 'ssp5_8_5').

    future_periods : tuple of tuples ((start_year_1, end_year_1), (start_year_2, end_year_2), ...)
        A list of tuples, each tuple representing the start and end years of a future period for which
        the deltas are to be calculated. Currently, only periods after 2014 are supported (data is 
        extracted from the 'future' experiment).

    output_directory : str
        The path to the directory where the output data is to be stored. If this directory does not exist, it
        will be created. If not provided, the data will be stored in the 'path_future_file' directory.

    """

    if path_future_file is None:
        raise ValueError("The path to the future data file must be provided.")

    if path_historical_file is not None:
        path_historical_file = path_future_file

    if output_directory is None:
        output_directory = path_future_file
    else:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)


    # Calculate the deltas
    for var in variables:

        historical_file = Path(path_historical_file).joinpath(f'{var}_historical.csv')
        future_file = Path(path_future_file).joinpath(f'{var}_{future_experiment}.csv')

        # Read the historical and truncate the data to the reference period
        historical = pd.read_csv(historical_file, index_col='date')
        historical = historical.truncate(before=f'{reference_period[0]}-01-01',
                                        after=f'{reference_period[1]}-12-31')

        # Read the future data. 
        future = pd.read_csv(future_file, index_col='date')

        # Find the common columns between the historical and future data
        common_columns = historical.columns.intersection(future.columns)

        with pd.ExcelWriter(f'{output_directory}/delta_{var}.xlsx') as writer:

            if var == 'tas':
                df = pd.DataFrame(historical[common_columns].mean(),
                                  columns=['mean (C)'], 
                                  index=common_columns).round(2)
                df.index.name = 'GCM'
                df.to_excel(writer, sheet_name='historical', columns=['mean (C)'])
            elif var == 'prcp':
                df = pd.DataFrame(historical[common_columns].mean(),
                                  columns=['mean (mm/day)'], 
                                  index=common_columns).round(2)
                df.index.name = 'GCM'
                df.to_excel(writer, sheet_name='historical', columns=['mean (mm/day)'])

            delta = pd.DataFrame(
                index=common_columns, 
                columns=[f'{future_periods[x][0]}-{future_periods[x][1]}' for x in range(len(future_periods))])
                                        
            if var == 'tas':
                for future_period in future_periods:
                    future_ = future.truncate(before=f'{future_period[0]}-01-01', after=f'{future_period[1]}-12-31')
                
                    delta[f'{future_period[0]}-{future_period[1]}'] = \
                        pd.DataFrame(future_[common_columns].mean() - historical[common_columns].mean()).round(2).values
                    
                delta.index.name = 'GCM'
                delta.to_excel(writer, sheet_name=f'Delta T (C) -- {future_experiment}')
                
            elif var == 'prcp':
                for future_period in future_periods:
                    future_ = future.truncate(before=f'{future_period[0]}-01-01', after=f'{future_period[1]}-12-31')
                    
                    delta[f'{future_period[0]}-{future_period[1]}'] = \
                        pd.DataFrame((future_[common_columns].mean() - historical[common_columns].mean()) \
                            / historical[common_columns].mean() * 100).round(2).values
                
                delta.index.name = 'GCM'   
                delta.to_excel(writer, sheet_name=f'Delta P (%) -- {future_experiment}')


def random_sampling_from_copula(future_periods, experiment, path_deltaT, path_deltaP, 
                                marginal_distributions=None, number_random_samples=1000, random_seed=42, 
                                output_directory=None, plot=True, xlim=None, ylim=None):
    
    """
    A function to generate random samples of the delta change in temperature and precipitation using a copula.
    Currently, the function supports only the Gaussian copula.

    Marginals are fitted to the delta change in temperature and precipitation. The best fitting distribution
    is selected based on the Kolmogorov-Smirnov test. The copula is then fitted to the delta change in temperature
    and precipitation. Random samples are generated using the fitted copula and the best fitting marginal distributions.

    Parameters
    ----------

    future_periods : list of tuples [(start_year_1, end_year_1), (start_year_2, end_year_2), ...]
        A list of tuples, each tuple representing the start and end years of a future period for which
        the deltas are to be calculated. The future periods must be available in the delta change data
        (typically generated from the 'calculate_delta_change' function).

    experiment : str
        The name of the experiment for which the deltas are to be calculated. The experiment must be available
        in the delta change data (typically generated from the 'calculate_delta_change' function).

    path_deltaT : str
        The path to the delta change in temperature file. The file must contain the delta change in temperature
        for the future periods and the experiment.

    path_deltaP : str
        The path to the delta change in precipitation file. The file must contain the delta change in precipitation
        for the future periods and the experiment.

    marginal_distributions : list of scipy.stats distributions
        A list of marginal distributions to be used to fit the delta change in temperature and precipitation.
        If None, the function will use the following distributions: [stats.norm, stats.expon, stats.gamma, stats.beta].

    number_random_samples : int (default is 1000)
        The number of random samples to generate.

    random_seed : int (default is the answer to the ultimate question of life, the universe, and everything.)
        The random seed to use when generating the random samples.

    output_directory : str
        The path to the directory where the output data is to be stored. If this directory does not exist, it
        will be created. If not provided, the data will be stored in the 'path_deltaT' directory.

    plot : bool (default is True)
        If True, a plot of the random samples and the GCMs will be created. 

    xlim : tuple (default is None)
        A tuple of two integers representing the lower and upper limits of the x-axis of the plot.

    ylim : tuple (default is None)
        A tuple of two integers representing the lower and upper limits of the y-axis of the plot.
       
    """

    if marginal_distributions is None:
        marginal_distributions = [stats.norm, stats.expon, stats.gamma, stats.beta]

    if output_directory is None:
        output_directory = os.path.dirname(path_deltaT)


    with pd.ExcelWriter(f'{output_directory}/random_delta_samples.xlsx') as writer:
    
        for future_start, future_end in future_periods:
        
            # Read the delta change in T and P
            deltaT = pd.read_excel(
                path_deltaT, sheet_name=f'Delta T (C) -- {experiment}', index_col='GCM')
            deltaP = pd.read_excel(
                path_deltaP, sheet_name=f'Delta P (%) -- {experiment}', index_col='GCM')

            # Find the GCMs that are common to both deltaT and deltaP
            common_gcms = list(set(deltaT.index) & set(deltaP.index))

            # Keep only the common GCMs
            deltaT = deltaT.loc[common_gcms]
            deltaP = deltaP.loc[common_gcms]

            # Create a new dataframe with the delta change in T and P
            delta_change = pd.DataFrame(index=deltaT.index, columns=['deltaP', 'deltaT'])
            delta_change['deltaT'] = deltaT[f'{future_start}-{future_end}']
            delta_change['deltaP'] = deltaP[f'{future_start}-{future_end}']

            # Fit a copula to the delta change in T and P
            # -------------------------------------------

            # Dictionary to store the results
            results = {}

            # Fit each distribution and calculate the goodness-of-fit
            marginals = {}
            for var in delta_change:
                for distribution in marginal_distributions:
                    params = distribution.fit(delta_change[var].values)
                    _, p_value = stats.kstest(delta_change[var].values, distribution.cdf, args=params)
                    results[distribution.name] = p_value

                # Store the best fitting distribution
                best_distribution_name = max(results, key=results.get)
                best_distribution_mod = [d for d in marginal_distributions if d.name == best_distribution_name][0]
                marginals[var] = {'name': best_distribution_name, 
                                'params': best_distribution_mod.fit(delta_change[var].values),
                                'pvalue': results[best_distribution_name], 
                                'model': best_distribution_mod}

            # Fit the copula
            # --------------
            copula = GaussianCopula()
            theta = copula.fit_corr_param(delta_change.values)

            joint_distribution = CopulaDistribution(
                copula=GaussianCopula(corr=theta),
                marginals=[
                    marginals['deltaP']['model'](*marginals['deltaP']['params']),
                    marginals['deltaT']['model'](*marginals['deltaT']['params'])],
                )
            samples = joint_distribution.rvs(number_random_samples, random_state=random_seed)

            # Save the samples
            samples = pd.DataFrame(samples, columns=['deltaP', 'deltaT'])
            samples.to_excel(writer, sheet_name=f'{future_start}-{future_end}', index=False)

            # Write the characteristics of the best fitting distributions
            df = pd.DataFrame(columns=['deltaP', 'deltaT'], index=['name', 'pvalue'])
            for var in marginals:
                df.loc['name', var] = marginals[var]['name']
                df.loc['pvalue', var] = marginals[var]['pvalue']
            df.to_excel(writer, sheet_name=f'Model {future_start}-{future_end}', index=True)
            

            if plot:

                if os.path.exists(f'{output_directory}/figures') == False:
                    os.makedirs(f'{output_directory}/figures')

                # Plot the samples
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(samples['deltaP'], samples['deltaT'], 'xk', label='Random samples')
                ax.plot(delta_change['deltaP'], delta_change['deltaT'], 'or', label='GCMs')
                ax.set_xlabel(r'$\Delta P (\%)$', fontsize=14)
                ax.set_ylabel(r'$\Delta T (C)$', fontsize=14)
                ax.set_title(f'Delta change factors ({future_start}-{future_end})', fontsize=16)
                ax.grid()
                plt.legend(fontsize=12)
                if xlim is not None:
                    ax.set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
                plt.tight_layout()
                fig.savefig(f'{output_directory}/figures/delta_change_{future_start}_{future_end}_w_random_samples.png')
                plt.show()


def plot_delta_change(path_deltaT, path_deltaP, future_exp, colors=None, color_map=None, xlim=None,
                      ylim=None, with_gcm_distribution_on_the_side=True, path_figure=None,
                      figure_title=None):
    
    """
        A function to plot the change in temperature and precipitation for different CMIP6 models.

        Parameters:
        -----------

        path_deltaT: str
            The path to the Excel file with the delta temperature data.

        path_deltaP: str
            The path to the Excel file with the delta precipitation data.

        future_exp: str
            The future experiment to plot.

        colors: list (optional)
            A list of colors to use for each period. If None, the 'YlGnBu' default colormap will be
            used. Rather than providing a list of colors, you can also provide the name of a colormap
            from matplotlib.cm.

        color_map: str (optional)
            The name of the colormap to use in case colors is None.            

        xlim: tuple (optional)
            The x-axis limits.

        ylim: tuple (optional)
            The y-axis limits.

        path_figure: str (optional)
            The path to save the figure. If None, the figure will be displayed.

        with_gcm_distribution_on_the_side: bool (optional)
            If True, the distribution of the GCMs will be plotted on the side of the main scatter plot.

        figure_title: str (optional)
            The title of the figure. If None, a default title will be used.

    
    """

    # Read the data
    deltaT = pd.read_excel(path_deltaT, sheet_name=f'Delta T (C) -- {future_exp}', index_col='GCM')
    deltaP = pd.read_excel(path_deltaP, sheet_name=f'Delta P (%) -- {future_exp}', index_col='GCM')

    # Find common models
    common_models = deltaT.index.intersection(deltaP.index)
    deltaT = deltaT.loc[common_models]
    deltaP = deltaP.loc[common_models]

    # Set the colors
    if colors is not None:
        if len(colors) != len(deltaT.columns):
            raise ValueError('The number of colors must be the same as the number of periods')
        else:
            colors = colors
    else:
        # Generate a list of colors from the 'YlGnBU' colormap. The list has the same length as the number of periods
        if color_map is not None:
            colors = plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(deltaT.columns)))
        else:
            colors = plt.cm.YlGnBu(np.linspace(0, 1, len(deltaT.columns)))

    if with_gcm_distribution_on_the_side == False:

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for k, period in enumerate(deltaT.columns):
            ax.scatter(deltaP[period], deltaT[period], s=120, c=colors[k], edgecolors='k', label=period,
                    zorder=2)
        ax.legend(fontsize=12)
        ax.set_ylabel(r'$\Delta T\ (C)$', fontsize=14)
        ax.set_xlabel(r'$\Delta P\ (\%)$', fontsize=14)
        if figure_title is not None:
            ax.set_title(figure_title, fontsize=16)
        else:
            ax.set_title(f'CMIP6 models ({future_exp})', fontsize=16)
        ax.grid(zorder=1)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        if path_figure is not None:
            fig.savefig(path_figure)
            plt.close()
        else:
            plt.show()

    elif with_gcm_distribution_on_the_side == True:


        if xlim is None or ylim is None:
            raise ValueError('You must provide the x and y limits to plot the GCM distribution on the'
            'side')
        
        if len(deltaT.columns) > 2:
            raise ValueError('You must provide maximum two periods to plot the GCM distribution on the side')
            
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(3, 3, width_ratios=[0.75, 2, 2], height_ratios=[2, 2, 0.75])

        ax1 = fig.add_subplot(gs[0:-1, 1:]) # Main scatter plot
        ax2 = fig.add_subplot(gs[2, 1:], sharex=ax1) # GCM distribution on the bottom
        ax3 = fig.add_subplot(gs[0:-1, 0], sharey=ax1) # GCM distribution on the side

        bin_width_P = 2.5
        bin_width_T = 0.5
        bins_P = np.arange(xlim[0], xlim[1], bin_width_P)
        bins_T = np.arange(ylim[0], ylim[1], bin_width_T)

        if len(deltaT.columns) == 1:
            alpha=[1]
        elif len(deltaT.columns) == 2:
            alpha=[1, 0.8]

        for k, period in enumerate(deltaT.columns):
            ax1.scatter(deltaP[period], deltaT[period], s=120, c=colors[k], edgecolors='k', label=period,
                    zorder=2)
            
            ax2.hist(deltaP[period].values.flatten(), bins=bins_P, color=colors[k], edgecolor='k', 
                     alpha=alpha[k], label=period)
            ax3.hist(deltaT[period].values.flatten(), bins=bins_T, color=colors[k], edgecolor='k', 
                     alpha=alpha[k], orientation='horizontal', label=period)
        ax1.legend(fontsize=12)
        ax1.set_ylabel(r'$\Delta T\ (C)$', fontsize=14)
        ax1.set_xlabel(r'$\Delta P\ (\%)$', fontsize=14)
        ax1.grid(zorder=1)
        if xlim is not None:
            ax1.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
        ax1.tick_params(axis='both', which='major', labelsize=12)

        
        ax2.set_xlabel(r'$\Delta P\ (\%)$', fontsize=14)
        ax2.set_ylabel('Nb of GCMs', fontsize=14)
        ax2.legend(fontsize=10)

        ax3.set_ylabel(r'$\Delta T\ (C)$', fontsize=14)
        ax3.set_xlabel('Nb of GCMs', fontsize=14)

        if figure_title is not None:
            plt.suptitle(figure_title, fontsize=16)
        else:
            plt.suptitle(f'CMIP6 models ({future_exp})', fontsize=16)

        
        plt.tight_layout()
        
        if path_figure is not None:
            fig.savefig(path_figure)
            plt.close()
        else:
            plt.show()


if '__main__' == __name__:

    
    # Extract basin average for each variable and experiment
    extract_basin_average_from_CMIP6_data(
        variables=['tas', 'prcp'], experiments=['historical', 'ssp5_8_5'], 
        basin_shapefile='../data/shapefiles/Orinoquia/cuencas_completas.shp',
        epsg=2317, # EPSG 2317 is a good projection to calculate areas for the Orinoco River basin (Colombia)
        gcm_directory='../data/CMIP6/CMIP6_zip/', 
        output_directory='../data/CMIP6/processed/')
    
    # Calculate the delta change
    calculate_delta_change(
        path_historical_file='../data/CMIP6/processed/',
        path_future_file='../data/CMIP6/processed/',
        variables=['tas', 'prcp'],
        future_experiment='ssp5_8_5',
        reference_period=(1985, 2014),
        future_periods=((2026, 2055), (2056, 2085)),
        output_directory='../data/CMIP6/processed/'
    )

    # Generate random samples from the copula
    random_sampling_from_copula(
        future_periods=((2026, 2055), (2056, 2085)),
        experiment='ssp5_8_5',
        path_deltaT = f'../data/CMIP6/processed/delta_tas.xlsx',
        path_deltaP = f'../data/CMIP6/processed/delta_prcp.xlsx',
        number_random_samples = 1000,
        random_seed = 42, # "The answer to the Ultimate Question of Life, the Universe, and Everything"
        output_directory = '../data/CMIP6/processed/',
        plot = True,
        xlim = (-42.5, 20),
        ylim = (0, 8),
    )

    # Plot the delta change
    plot_delta_change(path_deltaT='../data/CMIP6/processed/delta_tas.xlsx',
                    path_deltaP='../data/CMIP6/processed/delta_prcp.xlsx',
                    future_exp='ssp5_8_5',
                    colors=['#fee8c8', '#e34a33'],
                    xlim=(-42.5, 20),
                    ylim=(0, 8),
                    figure_title='CMIP6 models (SSP5-8.5) across Orinoquia Basin',
                    )
