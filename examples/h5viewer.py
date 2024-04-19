#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import yaml
import argparse

from bokeh.io import curdoc
from bokeh.models import DataTable, TableColumn, TextInput, ColumnDataSource, CustomJS
from bokeh.layouts import layout, column

parser = argparse.ArgumentParser(description='Run the Bokeh HDF5 Viewer.')
parser.add_argument('--path_hdf5', help='Path to the HDF5 file')
parser.add_argument('--port', help='Port to serve the application on', type=int, default=8080)
parser.add_argument('--websocket-origin', help='WebSocket origin', default='localhost:8080')
args = parser.parse_args()

# Function to recursively parse the HDF5 file structure and build a graph representation
def build_hdf5_graph(path_h5):
    graph = {}
    def process_node(h5_file_handle, node_name, parent_node_path):
        node_path = '/'.join([parent_node_path, node_name])
        node_val  = h5_file_handle[node_path]

        # A leaf node???
        if isinstance(node_val, h5py.Dataset):
            # Assign detailed data to leaf node...
            graph[node_path] = {
                'shape' : f'{node_val.shape}',
                'dtype' : f'{node_val.dtype}',
            }
        else:
            # Go through children nodes...
            for new_node_name in node_val.keys():
                process_node(h5_file_handle, new_node_name, node_path)

    # Handle data served over http as well.
    if path_h5.startswith("http://") or path_h5.startswith("https://"):
        # url = 'http://172.24.49.14:5000/fetch-data'
        url = path_h5
        # FIXME: specify these values from a url somehow
        payload = {
            'exp'          : exp,
            'run'          : run,
            'access_mode'  : access_mode,
            'detector_name': detector_name,
            'event'        : event
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            with io.BytesIO(response.content) as hdf5_bytes:
                with h5py.File(hdf5_bytes, 'r') as h5_file_handle:
                    process_node(h5_file_handle, node_name = '', parent_node_path = '')
            return data_array, pid, event
        else:
            print(f"Failed to fetch data from {url}: {response.status_code}")
    else:
        with h5py.File(path_h5, 'r') as h5_file_handle:
            process_node(h5_file_handle, node_name = '', parent_node_path = '')

    return graph

# Read the HDF5 file path from the command line
path_h5 = args.path_hdf5

# Build the graph
hdf5_graph = build_hdf5_graph(path_h5)


# [[[ Table ]]]
# Prepare data for Bokeh DataCube
source = ColumnDataSource(data=dict(
    name  = [ key          for key, val in hdf5_graph.items()],
    shape = [ val['shape'] for key, val in hdf5_graph.items()],
    dtype = [ val['dtype'] for key, val in hdf5_graph.items()],
))

## # Define target data source for DataCube selections
## target = ColumnDataSource(data=dict(row_indices=[], labels=[]))

# DataCube configuration
columns = [
    TableColumn(field='name' , title='Name' , width=400),
    TableColumn(field='shape', title='Shape', width=400),
    TableColumn(field='dtype', title='DType', width=800),
]

## grouping = [
##     GroupingInfo(getter='name'),
## ]

# Create the DataCube with the target
original_data = dict(source.data)
## cube = DataCube(source=source, columns=columns, grouping=grouping, target=target, width=2000, height=2000, sizing_mode='scale_width', fit_columns=True)
cube = DataTable(source=source, columns=columns, width=2000, height=2000, sizing_mode='scale_width', fit_columns=True)

# [[[ Search bar ]]]
# Create a TextInput widget to serve as the search bar
search_input = TextInput(value="", title="Search:")

# Callback function to filter the table based on the search input
def update_table(attr, old, new):
    # If the search bar is cleared, restore the original data
    if not new:
        source.data = original_data
    else:
        # Get the current value in the search bar
        search_value = new.strip().lower()
        # Create a new data dictionary with only the rows that match the search term
        new_data = {key: [] for key in original_data}
        for i in range(len(original_data['name'])):
            if search_value in original_data['name'][i].lower():
                for key in original_data:
                    new_data[key].append(original_data[key][i])
        # Update the data in the ColumnDataSource
        source.data = new_data

# Attach the update_table function to the search_input so that it triggers on text change
search_input.on_change('value', update_table)


## # CustomJS for filtering
## callback = CustomJS(args=dict(source=source), code="""
##     const original_data = source.data;
##     const search_value = this.value.trim().toLowerCase();
##     const new_data = {name: [], shape: [], dtype: []};
## 
##     if (search_value === '') {
##         // Restore the original data
##         Object.keys(new_data).forEach(key => {
##             new_data[key] = original_data[key].slice();
##         });
##     } else {
##         // Filter the data
##         for (let i = 0; i < original_data['name'].length; ++i) {
##             if (original_data['name'][i].toLowerCase().includes(search_value)) {
##                 new_data['name'].push(original_data['name'][i]);
##                 new_data['shape'].push(original_data['shape'][i]);
##                 new_data['dtype'].push(original_data['dtype'][i]);
##             }
##         }
##     }
##     source.data = new_data;
##     source.change.emit();
## """)
## 
## # Attach the CustomJS callback to the search_input so that it triggers on text change
## search_input.js_on_change('value', callback)

## # Layout
## l = layout([search_input, cube], sizing_mode='scale_both')

# Arrange the search bar and table in a layout
l = column(search_input, cube, sizing_mode='scale_both')

# Add the layout to the current document
curdoc().add_root(l)
