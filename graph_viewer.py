#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import numpy as np
import pandas as pd

from matplotlib import colormaps as cm
from matplotlib.colors import to_hex

import networkx as nx

from bokeh.io import show
from bokeh.plotting import figure

from bokeh.layouts import layout, column, row, grid
from bokeh.models import (
    BoxSelectTool,
    Circle,
    HoverTool,
    MultiLine,
    Plot,
    Range1d,
    ResetTool,
    ColumnDataSource,
    FixedTicker,
    LabelSet,
    TapTool,
    WheelZoomTool,
    PanTool,
    ColorBar,
    LinearColorMapper,
    BasicTicker,
    Button,
    CustomJS,
    MultiChoice,
    SaveTool,
    WheelZoomTool,
    Dropdown

)

from bokeh.palettes import linear_palette, Reds256
from bokeh.plotting import from_networkx, figure, curdoc

from bokeh.events import Tap, SelectionGeometry


# ## Read the graphs

# Each graph must be rapresented by an adjecency list (space separated)
# We assume nodes are numbered from 1 to N
#
# The list of points covereb by each node is a file with N lines, each line contains the points id (space separated)

# In[28]:


# ## Read the graphs from pickle
def read_graph_from_pickle(
    GRAPH_PATH, add_points_covered=False, MIN_SCALE=10, MAX_SCALE=25
):
    # read graph
    print("loading graph from pickle")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    try: 
        G.nodes
    except:
        G = G.Graph
        for node in G.nodes:
            G.nodes[node]['points_covered'] = G.nodes[node]['points covered'] 

    MAX_NODE_SIZE = 0
    for node in G.nodes:
        if len(G.nodes[node]["points_covered"]) > MAX_NODE_SIZE:
            MAX_NODE_SIZE = len(G.nodes[node]["points_covered"])

    for node in G.nodes:
        G.nodes[node]["label"] = str(node)

        G.nodes[node]["size"] = len(G.nodes[node]["points_covered"])
        # rescale the size for display
        G.nodes[node]["size rescaled"] = (
            MAX_SCALE * G.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SCALE
        )

        if not add_points_covered:
            del G.nodes[node]["points_covered"]

    # convert labels to int
    nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)}, copy=False)

    return G


def add_coloring(G, coloring_df, add_std=False):
    """ Takes pandas dataframe and compute the average and standard deviation \
    of each column for the subset of points colored by each ball.
    Add such values as attributes to each node in the BallMapper graph

    Parameters
    ----------
    coloring_df: pandas dataframe of shape (n_samples, n_coloring_function)

    add_std: bool, default=False
        Wheter to compute also the standard deviation on each ball.
    
    """
    # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
    for node in G.nodes:
        for name, avg in (
            coloring_df.loc[G.nodes[node]["points_covered"]].mean().items()
        ):
            G.nodes[node][name] = avg
        # option to add the standar deviation on each node
        if add_std:
            for name, std in (
                coloring_df.loc[G.nodes[node]["points_covered"]]
                .std()
                .items()
            ):
                if name in add_std:
                    G.nodes[node]["{}_std".format(name)] = std


def create_colorbar(style, palette, low, high):
    if style == "continuous":
        # continuous colorbar
        num_ticks = 100
        color_mapper = LinearColorMapper(
            palette=[
                to_hex(palette(color_id)) for color_id in np.linspace(0, 1, num_ticks)
            ],
            low=low,
            high=high,
        )

        return ColorBar(
            color_mapper=color_mapper,
            major_label_text_font_size="14pt",
            label_standoff=12,
        )

    elif style == "log":
        # log colorbar
        num_ticks = 100
        color_mapper = LogColorMapper(
            palette=[
                to_hex(palette(color_id)) for color_id in np.linspace(0, 1, num_ticks)
            ],
            low=low,
            high=high,
        )

        log_ticks = LogTicker(mantissas=[1, 2, 3, 4, 5], desired_num_ticks=10)

        return ColorBar(
            color_mapper=color_mapper,
            major_label_text_font_size="14pt",
            label_standoff=12,
            ticker=log_ticks,
        )

    elif style == "discrete":
        # discrete colorbar

        if var in ["signature", "s_mod3"]:
            step = 2
        else:
            step = 1

        ticks = [i for i in range(int(low), int(high) + 1, step)]

        color_mapper = LinearColorMapper(
            palette=[
                to_hex(palette(color_id)) for color_id in np.linspace(0, 1, len(ticks))
            ],
            low=low - step / 2,
            high=high + step / 2,
        )

        color_ticks = FixedTicker(ticks=ticks)

        return ColorBar(
            color_mapper=color_mapper,
            major_label_text_font_size="14pt",
            label_standoff=12,
            ticker=color_ticks,
        )


# function to color the nodes
# will be triggered each time SELECTED_NODES is updated
def color_selected_nodes(G, SELECTED_NODES=[]):
    for node in G.nodes:
        if node in SELECTED_NODES:
            G.nodes[node]["current_color"] = "black"
        else:
            G.nodes[node]["current_color"] = "white"


# Prepare Data
GRAPH1_PATH = sys.argv[1]

###########
# GRAPH 1 #
###########

# read graph
# ASSUME NODES ARE NUMBERED FROM 1 TO N
G = read_graph_from_pickle(GRAPH1_PATH, add_points_covered=True,
                        MAX_SCALE=20, MIN_SCALE=7)

print('{} nodes loaded'.format(len(G.nodes)))

# reading coloring df
print("reading coloring df")
coloring_df = pd.read_csv("data/coloring_df.csv")

if 'NA' in GRAPH1_PATH:
    coloring_df = coloring_df[coloring_df.is_alternating == 0]
    coloring_df.reset_index(inplace=True, drop=True)
    coloring_df.drop("is_alternating", axis=1, inplace=True)

coloring_df.drop("knot_id", axis=1, inplace=True)

coloring_df['s_signature_diff'] = np.abs(coloring_df['signature'].abs() -
                                         coloring_df['s_invariant'].abs())

add_std = ['signature', 's_invariant']
add_coloring(G, coloring_df, add_std=add_std)

## compute all colors
coloring_variables_dict = dict()
for var in list(coloring_df.columns) + ["{}_std".format(name) for name in add_std]:
    coloring_variables_dict[var] = dict()

# manually set each variable palette

# Here we adopt standard colour palette
my_palette = cm.get_cmap("jet")
my_red_palette = cm.get_cmap("Reds")

# coloring_variables_dict["size"]["palette"] = my_red_palette
# coloring_variables_dict["size"]["style"] = "log"

coloring_variables_dict["number_of_crossings"]["palette"] = my_red_palette
coloring_variables_dict["number_of_crossings"]["style"] = "continuous"

if 'NA' not in GRAPH1_PATH:
    coloring_variables_dict["is_alternating"]["palette"] = my_red_palette
    coloring_variables_dict["is_alternating"]["style"] = "continuous"

coloring_variables_dict["s_invariant"]["palette"] = my_palette
coloring_variables_dict["s_invariant"]["style"] = "discrete"

coloring_variables_dict["signature"]["palette"] = my_palette
coloring_variables_dict["signature"]["style"] = "discrete"

coloring_variables_dict["signature_mod4"]["palette"] = my_red_palette
coloring_variables_dict["signature_mod4"]["style"] = None

coloring_variables_dict["s_signature_diff"]["palette"] = my_red_palette
coloring_variables_dict["s_signature_diff"]["style"] = "continuous"

for var in ["{}_std".format(name) for name in add_std]:
    coloring_variables_dict[var]["palette"] = my_red_palette
    coloring_variables_dict[var]["style"] = "continuous"


for var in coloring_variables_dict:
    print(var)
    MIN_VALUE = 10000
    MAX_VALUE = -10000

    for node in G.nodes:
        if G.nodes[node][var] > MAX_VALUE:
            MAX_VALUE = G.nodes[node][var]
        if G.nodes[node][var] < MIN_VALUE:
            MIN_VALUE = G.nodes[node][var]

    coloring_variables_dict[var]["max"] = MAX_VALUE
    coloring_variables_dict[var]["min"] = MIN_VALUE

    for node in G.nodes:
        if not pd.isna(G.nodes[node][var]):
            color_id = (G.nodes[node][var] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
            if coloring_variables_dict[var]["style"] == "log":
                color_id = (np.loG0(G.nodes[node][var]) - np.loG0(MIN_VALUE)) / (
                    np.loG0(MAX_VALUE) - np.loG0(MIN_VALUE)
                )
            G.nodes[node]["{}_color".format(var)] = to_hex(
                coloring_variables_dict[var]["palette"](color_id)
            )
        else:
            G.nodes[node]["{}_color"] = "black"

for node in G.nodes:
    G.nodes[node]["current_color"] = "white"

# save memory
for node in G.nodes:
    del G.nodes[node]["points_covered"]


SELECTED_NODES = []

# ## UI

##########
#  PLOT  #
##########

print('rendering graph')

plot = Plot(
    width=800,
    height=600,
    x_range=Range1d(-1.1, 1.1),
    y_range=Range1d(-1.1, 1.1),
)

node_hover_tool = HoverTool(tooltips=[("id", "@label"), ("size", "@size")])
plot.add_tools(
    PanTool(),
    node_hover_tool,
    BoxSelectTool(),
    WheelZoomTool(),
    ResetTool(),
    TapTool(),
    SaveTool(),
)

zoom_tool = WheelZoomTool()
plot.toolbar.active_scroll = zoom_tool

graph_renderer = from_networkx(
    graph=G,
    layout_function = nx.spring_layout,
    seed=42,
    scale=1,
    center=(0, 0),
    k=10 / np.sqrt(len(G.nodes)),
    iterations=2000,
)

## labels
# get the coordinates of each node
x_1, y_1 = zip(*graph_renderer.layout_provider.graph_layout.values())

# create a dictionary with each node position and the label
source_1 = ColumnDataSource(
    {
        "x": x_1,
        "y": y_1,
        "node_id": [node["label"] for _, node in G.nodes(data=True)],
    }
)
labels_1 = LabelSet(
    x="x", y="y", text="node_id", source=source_1, text_color="black", text_alpha=0
)

# nodes
graph_renderer.node_renderer.glyph = Circle(
    size="size rescaled", fill_color="current_color", fill_alpha=0.8
)

# edges
graph_renderer.edge_renderer.glyph = MultiLine(
    line_color="black", line_alpha=0.8, line_width=1
)

plot.renderers.append(graph_renderer)
plot.renderers.append(labels_1)

################
#  color bars  #
################

color_bar_dict = {}

for var in coloring_variables_dict:
    if coloring_variables_dict[var]["style"]:
        color_bar_dict[var + "_color"] = create_colorbar(
            style=coloring_variables_dict[var]["style"],
            palette=coloring_variables_dict[var]["palette"],
            low=coloring_variables_dict[var]["min"],
            high=coloring_variables_dict[var]["max"],
        )
        color_bar_dict[var + "_color"].visible = False
        color_bar_dict[var + "_color"].title = var.replace("_", " ")
        color_bar_dict[var + "_color"].title_text_font_size = "14pt"

for key in color_bar_dict:
    plot.add_layout(color_bar_dict[key], "right")

###################
#  dropdown menu  #
###################
code = """ 
        
        var node_data = graph_renderer.node_renderer.data_source.data;
        var edge_data = graph_renderer.edge_renderer.data_source.data;
        for (var i = 0; i < node_data['size'].length; i++) {
            
            graph_renderer.node_renderer.data_source.data['current_color'][i] = node_data[this.item][i];
        }
        
        
        for (var key in color_bar_dict){
            color_bar_dict[key].visible = false;
        }
        
        if (this.item in color_bar_dict) {
            color_bar_dict[this.item].visible = true;

        }
        
        graph_renderer.node_renderer.data_source.change.emit();
        graph_renderer.edge_renderer.data_source.change.emit();
        
        for (var key in color_bar_dict){
            color_bar_dict[key].change.emit();
        }


    """

callback = CustomJS(
    args=dict(graph_renderer=graph_renderer, color_bar_dict=color_bar_dict),
    code=code,
)

menu = [(var.replace("_", " "), var + "_color") for var in coloring_variables_dict]

dropdown = Dropdown(
    label="Select a coloring function", button_type="default", menu=menu,
    sizing_mode='stretch_width'
)
dropdown.js_on_event("menu_item_click", callback)

################
# print button #
################

save_button = Button(
    label="SAVE SELECTED", sizing_mode='stretch_width', button_type="success"
)

#################
# labels button #
#################

labels_button = Button(
    label="SHOW LABELS",
    sizing_mode='stretch_width',
)
# button_type="success")

def showLabel():
    if labels_button.label == "HIDE LABELS":
        labels_button.label = "SHOW LABELS"
        labels_1.text_alpha = 0
    else:
        labels_button.label = "HIDE LABELS"
        labels_1.text_alpha = 1

labels_button.on_click(showLabel)

###################
# multichoice box #
###################

OPTIONS = [n["label"] for _, n in G.nodes(data=True)]

multi_choice = MultiChoice(value=[], options=OPTIONS, sizing_mode='stretch_width', height=50)
# multi_choice.js_on_change("value", CustomJS(code="""
#     console.log('multi_choice: value=' + this.value, this.toString())
# """))

# # this function is called when the MultiChoice object is modified
# def update():
#     SELECTED_NODES = [int(n) for n in multi_choice.value]
#     color_nodes(G, SELECTED_NODES)
#     graph_renderer.node_renderer.data_source.data['color'] = [G.nodes[n]['color'] for n in G.nodes]

def save_nodelist():
    print("\nThe selected nodes are: ")
    print(set([n for n in multi_choice.value]))
    with open('{}_selected_nodes.pkl'.format(GRAPH1_PATH[:-4]), 'wb') as f:
        pickle.dump(list(set([n for n in multi_choice.value])), f)

# color_button.on_click(update)

save_button.on_click(save_nodelist)

taptool = plot.select(type=TapTool)

def update_node_highlight(event):
    nodes_clicked = graph_renderer.node_renderer.data_source.selected.indices
    multi_choice.value += [G.nodes[n]["label"] for n in nodes_clicked]

    SELECTED_NODES = [
        n for n in G.nodes if G.nodes[n]["label"] in multi_choice.value
    ]

    color_selected_nodes(G, SELECTED_NODES)
    graph_renderer.node_renderer.data_source.data["current_color"] = [
        G.nodes[n]["current_color"] for n in G.nodes
    ]

plot.on_event(Tap, update_node_highlight)
plot.on_event(SelectionGeometry, update_node_highlight)

##########
# LAYOUT #
##########
layout = grid([[dropdown, labels_button, save_button], [multi_choice], [plot]], 
               sizing_mode="stretch_width"
)

# layout = column(row(button, multi_choice),
#                 row(plot, plot2, sizing_mode="stretch_both"),
#                 )

curdoc().add_root(layout)

print('done')
