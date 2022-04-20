"""
Download historic ERA5 files
Initial created by Matthew King 2019 @ NRC
Notes
-----
Get your UID and API key from the CDS portal at the address
https://cds.climate.copernicus.eu/user and write it into
the configuration file, so it looks like:
$ cat ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API key>
verify: 0
References
----------
https://pypi.org/project/cdsapi/
"""

import argparse
import cdsapi
import os

def download_era5(output_dir, url="https://cds.climate.copernicus.eu/api/v2", bbox=(90, -180, 40, 180), start_year=1979, end_year=2018, features=None):
    """
    Download ERA5 files
    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    # output_dir = os.getcwd()

    base = "ERA5_"

    if features is None:
        features = ['10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    '10m_wind_gust_since_previous_post_processing',
                    '2m_dewpoint_temperature', '2m_temperature',
                    'evaporation',
                    'mean_sea_level_pressure', 'mean_wave_direction',
                    'mean_wave_period',
                    'sea_ice_cover', 'sea_surface_temperature',
                    'significant_height_of_combined_wind_waves_and_swell',
                    'snowfall', 'snowmelt', 'surface_latent_heat_flux',
                    'surface_sensible_heat_flux', 'total_cloud_cover',
                    'total_precipitation',
                    'surface_solar_radiation_downwards']

    for year in range(start_year, end_year):
        print(year)
        os.chdir(output_dir)

        for month in range(1, 13):  # up to 12
            os.chdir(output_dir)

            print(month)
            # '01' instead of '1'
            month = str(month).rjust(2, '0')

            # eg. 1979-01
            subdirectory = "{}-{}".format(year, month)
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)

            os.chdir(subdirectory)

            # _197901.nc
            extension = "_{}{}.nc".format(year, str(month).rjust(2, '0'))

            for feature in features:
                print(feature)

                # eg. ERA5_10m_u_component_of_wind_197901.nc
                filename = base + feature + extension

                if not os.path.isfile(filename):
                    print("Downloading file {}".format(filename))

                    downloaded = False

                    while not downloaded:
                        try:
                            client = cdsapi.Client(url=url, retry_max=5)
                            client.retrieve(
                                'reanalysis-era5-single-levels',
                                {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'variable': feature,
                                    'area': [str(b) for b in bbox],
                                    'time': [
                                        '00:00', '03:00', '06:00',
                                        '09:00', '12:00', '15:00',
                                        '17:00', '21:00'
                                    ],
                                    'day': [
                                        '01', '02', '03',
                                        '04', '05', '06',
                                        '07', '08', '09',
                                        '10', '11', '12',
                                        '13', '14', '15',
                                        '16', '17', '18',
                                        '19', '20', '21',
                                        '22', '23', '24',
                                        '25', '26', '27',
                                        '28', '29', '30', '31'
                                    ],
                                    # API ignores cases where there are less than 31 days
                                    'month': month,
                                    'year': str(year)
                                },
                                filename)

                        except Exception as e:
                            print(e)

                            # Delete the partially downloaded file.
                            if os.path.isfile(filename):
                                os.remove(filename)

                        else:
                            # no exception implies download was complete
                            downloaded = True


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Download ERA5 files to current working directory.")
    # parser.add_argument("output_dir", help='directory to store the data')
    # parser.add_argument("key",help='portal account key')
    # parser.add_argument("url",help='download portal url',default="https://cds.climate.copernicus.eu/api/v2")
    # parser.add_argument("start_year", type=int, default=1979,
    #                     help="Year to start in YYYY format.")
    # parser.add_argument("end_year", type=int, default=2018,
    #                     help="Year to end in YYYY format.")

    # args = parser.parse_args()

    # download_era5(output_dir=args.output_dir, key=args.key, url=args.url, start_year=args.start_year, end_year=args.end_year)

    download_era5('/Users/Zach/OneDrive - University of Waterloo/Documents/MASc/Courses/SYDE770/project/polynya-prediction/test',
        url="https://cds.climate.copernicus.eu/api/v2",
        start_year=2002, end_year=2005,
        features=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature'])