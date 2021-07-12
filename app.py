import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from simulation import *
import matplotlib.pyplot as plt

"""
# Network Epidemic Simulator
Continuous-time stochastic simulation of epidemic spreading on human-to-human contact networks:
"""

'''
## Epidemic Model
'''
states = ["S", "I", "R"]
rules = [("I", "R", 1.0), # spontaneous rule  I -> R with rate 1.0
        ("R", "S", 0.7),  # spontaneous rule R -> S with rate 0.7
        (("I","S"),("I","I"), 0.8)]
rules_text = [repr(r) for r in rules]
rules_text = '\n'.join(rules_text)

epidemic_states_text = st.text_input("Enter states/compartments (separated by comma):", value='S,I,R')
epidemic_model_text = st.text_area("Enter interaction rules (separated by line):", value=rules_text)

'''
## Contact Network
'''
edgelist_text = ([(0, 4), (0, 1), (1, 5), (1, 2), (2, 6), (2, 3), (3, 7), (4, 8), (4, 5), (5, 9), (5, 6), (6, 10), (6, 7), (7, 11), (8, 12), (8, 9), (9, 13), (9, 10), (10, 14), (10, 11), (11, 15), (12, 13), (13, 14), (14, 15)])
edgelist_text = [repr(e) for e in edgelist_text]
edgelist_text = ','.join(edgelist_text)
network_text = st.text_area("Enter contact networks as edgelist:", value=edgelist_text)

# network eval
network_eval = eval('[' + network_text + ' ]')
import networkx as nx
G = nx.from_edgelist(network_eval)
G = nx.convert_node_labels_to_integers(G)

#write graph info
st.write(str(nx.info(G)))
if G.number_of_nodes() < 201:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.scatter([1,2,3],[1,2,3])
    nx.draw(G, pos=nx.spectral_layout(G), node_color='black', edgelist=list(), alpha=0.5, linewidths=0.0)
    nx.draw(G, pos=nx.spectral_layout(G), node_color='black', nodelist=list())
    st.write(fig, width=0.5)

'''
## Initial State
'''

# eval

states_eval = eval('"' + epidemic_states_text.replace(',','","')+'"')
#st.write(str(G.nodes))
node_num = G.number_of_nodes()
rules_eval = eval('[' + epidemic_model_text.replace('\n',',') + ' ]')

states_slider = dict()
for s in states_eval:
    x = st.slider('Init Number '+str(s), min_value=0.0, max_value=float(node_num), value=1.0,step=1.0)
    states_slider[s] = x

slider_sum = np.sum(list(states_slider.values()))
states_slider_normal = dict()
for s, v in states_slider.items():
    states_slider_normal[s] = int(v/slider_sum * G.number_of_nodes())
slider_sum_normal = np.sum(list(states_slider_normal.values()))
s0 = states_eval[0]
slider_sum_normal_expects1 = slider_sum_normal - states_slider_normal[s0]
states_slider_normal[s0] = int(G.number_of_nodes() - slider_sum_normal_expects1)

'''
## Run Simulation
'''
horizon_slider = st.slider('Horizon', min_value=1.0, max_value=200.0, value=20.0)

click_run = st.button('Run Simulation')



if click_run or 'chart_data' in st.session_state:
    if click_run:
        chart_data, label_data = run_simulation(G, states_eval, rules_eval, states_slider_normal, horizon_slider)
        st.session_state['chart_data'] = chart_data  # Dictionary like API
        st.session_state['horizon'] = horizon_slider
        st.session_state['label_data'] = label_data
    else:
        chart_data = st.session_state['chart_data']
        label_data = st.session_state['label_data']
    #st.write(str(states_eval))
    #st.write(str(G.nodes))
    #st.write(str(G.edges))
    #st.write(str(rules_eval))
    #st.write(repr(states_slider))
    #st.write(slider_sum)
    st.line_chart(chart_data)
    #st.write(repr(states_slider_normal))
    #st.write(horizon_slider)
    if 'ct_value' in st.session_state:
        ct_value = st.session_state.ct_value
    else:
        ct_value = 0.0
    current_time_slider = st.slider('Timepoint', min_value=0.0, max_value=st.session_state['horizon'], value=0.0, step=0.1)
    st.session_state['ct_value'] = current_time_slider


    chart_data_filtered = chart_data.reset_index()
    chart_data_filtered['simple_count'] = list(range(len(chart_data_filtered['time'])))
    #st.write(chart_data_filtered.head(30))
    #st.write(current_time_slider)
    chart_data_filtered = chart_data_filtered[chart_data_filtered.time <= float(current_time_slider)]

    current_row = chart_data_filtered.tail(1)
    #st.write(current_row)
    cuttent_df_id = int(current_row['simple_count'])
    current_labels = label_data[cuttent_df_id]
    #st.write(current_labels)
    current_labels_dict = {n: current_labels[n] for n in range(len(current_labels))}

    if G.number_of_nodes() < 201:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        colordict = {'S': 'yellow', 'I':'red', 'R':'green'}
        colorlist = [colordict.get(l, 'black') for l in current_labels]
        nx.draw(G, pos=nx.spectral_layout(G), edgelist=list(), alpha=1.0, linewidths=0.0, labels=current_labels_dict, font_color='gray', node_color = colorlist)
        nx.draw(G, pos=nx.spectral_layout(G), node_color='black', nodelist=list())
        st.write(fig, width=0.5)


#
# st.graphviz_chart('''
#     graph {
#         run -- intr
#         intr -- runbl
#         runbl -- run
#     }
# ''')
#
# #import pygraphviz
# #from networkx.drawing.nx_agraph import write_dot
# #write_dot(G, "grid.dot")
#
#
#
# import streamlit as st
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.scatter([1,2,3],[1,2,3])
# st.write(fig)
