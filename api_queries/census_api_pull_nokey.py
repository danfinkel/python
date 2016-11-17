# -*- coding: utf-8 -*-
"""
Community Profile ACS Download Tool
Original author: Michael Silva
Originally started on Tue Apr 21 15:54:31 2015

Modified by Dan Finkel
Most recent updates: November 2016

Description: This script downloads ACS data for US by zip code (all 33K rows)
    The ACS is a annual US census survey of 3.5 million addresses (a complement
    to the decennial survey)
"""
import requests
import pandas
import numpy as np

# My API key
api_key = 'COPY YOUR CENSUS KEY HERE'


def download_acs_data(var_list, api_key, api_url_base, st_idx, end_idx):
    """
    Download census acs data
        - Build url and call using pandas
        - Rename columns to accurately describe variables
        - Return data in pandas dataframe
    """

    # Extract variable information
    get_string = ''
    labels = list()
    for variable in var_list[st_idx:end_idx]:
        get_string = get_string + ',' + variable[0]
        labels.append(variable[1])

    # Construct and call url
    api_url = api_url_base + get_string + '&for=zip+code+tabulation+area:*' + '&key=' + api_key
    zcta_data = pandas.io.json.read_json(api_url)

    # Rename columns based on first row
    zcta_data.columns = zcta_data[:1].values.tolist()
    zcta_data.columns = ['Name'] + labels + ['zip_code']

    # Drop first row
    return zcta_data[1:]


if __name__ == '__main__':

    # census year pull
    year = 2013

    # table
    # The B01001 table is a summary dataset
    # Other tables are available and segment data, e.g., by race
    table = 'B01001'

    # get acs variable list
    variables_url = 'http://api.census.gov/data/' + str(year) + '/acs5/variables.json'
    data = requests.get(url=variables_url)
    data = data.json()

    # Pull variables in the table
    var_list = [(v, data['variables'][v]['label']) for v in data['variables'] if v.split('_')[0] == table]

    # Download the data
    api_url_base = 'http://api.census.gov/data/' + str(year) + '/acs5?get=NAME'

    # 50 variable maximum per api call, so we need to split up
    data = []
    total_calls = int(np.floor(len(var_list)) / 49.)
    for iteration in np.arange(total_calls):
        st_idx = iteration * 49
        end_idx = (iteration + 1) * 49
        data.append(download_acs_data(var_list, api_key, api_url_base, st_idx, end_idx))

    # Concat data and save to disk
    out_data = pandas.concat(data, axis=1)
    out_data.to_excel('zip_data.xlsx')
