

import os
from itertools import product
import cdsapi
import pandas as pd
import geopandas as gpd

def from_ClimateDataStore(GCMs=None, variables=None, domain=None, period=(1980, 2101), 
                            future_experiments=None, temporal_resolution='monthly', 
                            list_of_months=None, output_dir=None, overwrite=False):
    
    """ Download CMIP6 data from the Copernicus Data Store (CDS) using the cdsapi package."
        https://cds.climate.copernicus.eu/#!/home

        Parameters:
        -----------
        
        period : tuple
            Period over which to download the data. Default is (1980, 2101).
        
        GCMs : list
            List of GCMs to download data from. Default is None, in which case data from all GCMs 
            will be downloaded.  See list 'models' for available GCMs.

        variables : tuple of tuples
            List of variables to download. Default is None, in which case data for tas and prcp 
            will be downloaded. Each tuple should contain the a short variable name and the variable
            long name used in CDS. The short variable name will be used to name the downloaded files.

        domain : list, tuple or GeoDataFrame
            Domain over which to download the data. If a list or tuple, it should contain the 
            coordinates of the domain in the following order: [latN, lonW, latS, lonE]. If a 
            GeoDataFrame, the domain will be the total bounds of the GeoDataFrame. 
            If None, the global domain will be used.

                
        future_experiments : str or list of str
            Experiment to download the data for.  If a list of experiments is provided, the data for all 
            experiments will be downloaded.
            Acceptable values are: ['ssp1_1_9', 'ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp4_3_4', 
                                   'ssp4_6_0', 'ssp5_8_5',  'ssp5_3_4os', 'ssp5_8_5']

            If None and the period[1]<2015, only the historical experiment will be downloaded. 
            If None and the period[0]>=2015, the script will raise an error and ask the user to provide
            at least one future experiment to download the data for.

        temporal_resolution : str
            Temporal resolution of the data to download. Default is 'monthly'. Other options are 'daily'
            and 'hourly'. The temporal resolution will be used to name the downloaded files.

            Note: Currently, 'monthly' is the only temporal resolution that is supported.

        list_of_months : list (Optional)
            List of months to download the data for. Default is None, in which case the data for all 
            months will be downloaded.

        output_dir : str
            Directory where the downloaded files will be saved. If None, an error will be raised 
            prompting the user to provide an output directory.


    """
        

    # Check if the user has a CDS API key
    home_dir = os.path.expanduser('~')
    if not os.path.exists(f'{home_dir}/.cdsapirc'):
        raise ValueError('Please create a .cdsapirc file in your home directory with your CDS API key'
                         '(https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows)')
    
    # Check if the user has the cdsapi package installed
    try:
        import cdsapi
    except ImportError:
        raise ImportError('Please install the cdsapi package by running "pip install cdsapi"')
    
    # Check if the API key is valid
    try: 
        c = cdsapi.Client()
    except Exception as e:
        raise ValueError(f'Error: {e}')

    
    if GCMs is None:
    
        # Default list of GCMs that will be downloaded
        models = ('access_cm2', 'access_esm1_5', 'awi_cm_1_1_mr', 'awi_esm_1_1_lr', 'bcc_csm2_mr',
            'bcc_esm1', 'cams_csm1_0', 'canesm5', 'canesm5_canoe', 'cesm2', 'cesm2_fv2', 'cesm2_waccm',
            'cesm2_waccm_fv2', 'ciesm', 'cmcc_cm2_hr4', 'cmcc_cm2_sr5', 'cmcc_esm2', 'cnrm_cm6_1', 
            'cnrm_cm6_1_hr',  'cnrm_esm2_1', 'e3sm_1_0', 'e3sm_1_1', 'e3sm_1_1_eca', 'ec_earth3', 
            'ec_earth3_aerchem', 'ec_earth3_cc', 'ec_earth3_veg', 'earth3_veg_lr', 'fgoals_f3_l', 
            'fgoals_g3', 'fio_esm_2_0', 'gfdl_esm4', 'giss_e2_1_g', 'giss_es2_1_h', 'hadgem3_gc31_ll',
            'hadgem3_gc31_mm', 'iitm_esm', 'inm_cm4_8', 'inm_cm5_0', 'ipsl_cm5a2_inca', 'ipsl_cm6a_lr',
            'kace_1_0_G', 'kiost_esm', 'mcm_ua_1_0', 'miroc6', 'miroc_es2h', 'miroc_es2l', 
            'mpi_esm1_1_2_ham', 'mpi_esm1_2_hr', 'mpi_esm1_2_lr', 'mri_esm2_0', 'nesm3', 'norcpm1', 
            'noresm2_lm', 'noresm2_mm', 'sam0_unicon', 'taiesm1', 'ukesm1_0_ll')
        
    if variables is None:
        
        # Default list of variables that will be downloaded
        variables = (
            ('tas', 'near_surface_air_temperature'),
            ('prcp', 'precipitation')
        )

    # Define the domain over which to download the data
    if domain is None:
        latN, lonW, latS, lonE = [90, -180, -90, 180] # Global domain
        Warning('No domain was specified. Downloading data for the global domain')

    elif type(domain) == list or type(domain) == tuple:
        if len(domain) != 4:
            raise ValueError('The domain should be a list or tuple of 4 elements: [latN, lonW, latS, lonE]')
        else:
            latN, lonW, latS, lonE = domain
                
    elif type(domain) == gpd.geodataframe.GeoDataFrame:
        latN, lonW, latS, lonE = domain.total_bounds.tolist()

    else:
        raise ValueError('The domain should be a list, tuple or GeoDataFrame')
    
    # Define the years over which to download the data
    start_year, end_year = period
    if start_year > end_year:
        raise ValueError('The start year should be smaller than the end year')
    
    if end_year > 2015 and future_experiments == None:
        raise ValueError('Please provide at least one future experiment to download the data for')
    
    if end_year < 2015:
        experiment = 'historical'

    if end_year > 2100:
        end_year = 2100
        raise Warning('The end year was set to 2100')
        
    if start_year >= 2015:
        experiment = future_experiments

    if start_year < 2015 and end_year >= 2015:
        experiment = ['historical', future_experiments]

    if temporal_resolution != 'monthly':
        raise ValueError('Currently, only the monthly temporal resolution is supported')
    
    # Define the months to download the data for
    if list_of_months is None:
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    else:
        months = [str(i) for i in list_of_months]

    if output_dir is None:
        raise ValueError('Please provide an output directory to save the downloaded files')


    # Download the data
    list_failed_requests = pd.DataFrame(
        columns=['temporal_resolution', 'experiment', 'variable', 'model'])
    for model, variable, exp in product(models, variables, experiment):

        short_name, long_name = variable     
        

        if overwrite is False:
            expected_output = f'{output_dir}/CMIP6/{short_name}/{exp}/{model}_{short_name}.zip'
            if os.path.exists(expected_output):
                print(f'The file {expected_output} already exists. Skipping...')
                continue
        
        # Create directories to save the downloaded files
        full_output_dir = f'{output_dir}/CMIP6/{short_name}/{exp}/'
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        
        dataset = 'projections-cmip6'

        # Create the vector of years to download the data for
        if exp == 'historical':
            if end_year > 2014:
                years = [str(i) for i in range(start_year, 2015)]
            else:
                years = [str(i) for i in range(start_year, end_year+1)]

        else:
            if start_year < 2015:
                years = [str(i) for i in range(2015, end_year+1)]
            else:
                years = [str(i) for i in range(start_year, end_year+1)]

        request = {                
            'temporal_resolution': temporal_resolution,
            'experiment': exp,
            'variable': long_name,
            'model': model,
            'year': years,
            'month': months,
            'area': [latN, lonW, latS, lonE],
            'format': 'zip'
        }

        #print(request)
        
        try:
            client = cdsapi.Client()
            client.retrieve(dataset, request, f'{full_output_dir}/{model}_{short_name}.zip')
        except Exception as e:
            print(f'Error: {e}')
            list_failed_requests._append(
                {'temporal_resolution': temporal_resolution,
                 'experiment': exp,
                 'variable': short_name,
                 'model': model}, 
                 ignore_index=True)
            continue
        else:
            print(f'Data for {model} and {short_name} was downloaded successfully')

        del client

    print('All data was downloaded successfully')

    list_failed_requests.to_csv(f'{output_dir}/CMIP6/failed_requests.csv', index=False)