# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:58:01 2020

@author: terry
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import random
seed = 1
from scipy.spatial import distance
from itertools import combinations



class City:
    graph = g = nx.Graph()
    node_positions = []
    building_nodes = []
    edges = []  # only used in generate_city()
    nodes = []  # only used in generate_city()

def coordinates_nodes(G,node):
    return list(G.nodes[node].values())[:2]

def dist_nodes(G,A,B):
    return distance.cityblock(coordinates_nodes(G,A),coordinates_nodes(G,B))

def generate_city(n_prob,b_prob):
    node_prob = n_prob
    building_prob = b_prob

    edgelist = pd.read_csv('data/edge.csv')
    nodelist = pd.read_csv('data/nodes.csv')

    g = nx.Graph()

    for i, element in edgelist.iterrows():
        g.add_edge(element[0], element[1], weight=random.randint(5, 50), color=element[3])

    for i, element in nodelist.iterrows():
        g.nodes[element['id']].update(element[1:].to_dict())
        
    # for i, element in edgelist.iterrows():
    #     g.add_edge(element[0], element[1], weight=random.randint(5, 50), color=element[3])
        # distance = dist_nodes(g, element[0], element[1])


    # delete some nodes randomly
    for node in list(g.nodes):
        if node == 'n00' or node == 'n44':
            continue
        else:
            if random.random() < node_prob:
                t = deepcopy(g)
                t.remove_node(node)
                if nx.is_connected(t):
                    g.remove_node(node)
                t.clear()

    x_coor = []
    y_coor = []

    for node in list(g.nodes):
        x_coor.append(g.nodes[node]['x'])
        y_coor.append(g.nodes[node]['y'])

    nodemap = {'id': list(g.nodes), 'x': x_coor, 'y': y_coor}

    # add buildings randomly
    building_nodes = []
    i = 1
    for node in list(g.nodes):
        dic = {}
        if g.nodes[node]['x'] == 0 or g.nodes[node]['x'] == 900 or g.nodes[node]['y'] == 0 or g.nodes[node]['y'] == 900:
            continue
        else:
            if random.random() < building_prob or i == 0:
                dic['id'] = chr(ord('@') + i)
                dic['x'] = g.nodes[node]['x'] + 50
                dic['y'] = g.nodes[node]['y']
                dic['num_bikes'] = random.randint(10,80)
                building_nodes.append(dic)
                i = i + 1

                g.add_node(dic['id'], x=dic['x'], y=dic['y'])
                g.add_edge(node, dic['id'], weight=random.randint(5, 50), color='black')

                adj_node = [x for x, y in g.nodes(data=True) if y['x'] == g.nodes[node]['x'] + 100 and y['y'] == g.nodes[node]['y']]
                for n in adj_node:
                    g.remove_edge(node, n)
                    g.add_edge(n, dic['id'], weight=random.randint(5, 50), color='black')

    build_id = []
    build_x = []
    build_y = []
    build_bikes = []
    for build in building_nodes:
        build_id.append(build['id'])
        build_x.append(build['x'])
        build_y.append(build['y'])
        build_bikes.append(build['num_bikes'])

    building = {'id': build_id, 'x': build_x, 'y': build_y, 'num_bikes': build_bikes }

    buildingnodes = pd.DataFrame(building, columns=['id', 'x', 'y','num_bikes'])

    all_nodes = list(g.nodes)
#     print(all_nodes)
    # Creat a list to store all distances between nodes
    dist_dict = {}
    for nodes in list(combinations(all_nodes, 2)):
        node = list(nodes)
        dist_dict[(node[0], node[1])]=dist_nodes(g,node[0], node[1])
    nx.set_edge_attributes(g,dist_dict,'distance')
    edgenode = []
    weight = []
    dist = []
    for edge in list(g.edges):
        edgenode.append(edge)
        weight.append(g.edges[edge]['weight'])
        dist.append(g.edges[edge]['distance'])

    left = []
    right = []

    for item in edgenode:
        left.append(item[0])
        right.append(item[1])

    edgenodes = {'node1': left, 'node2': right, 'weight': weight,'distance': dist}

    ednodes = pd.DataFrame(edgenodes, columns=['node1', 'node2', 'weight','distance'])

    city = City()
    city.building_nodes = buildingnodes
    city.edges = ednodes
    city.nodes = pd.DataFrame(nodemap, columns=['id', 'x', 'y'])
    city.graph = g


    return city

def random_select(nodes_list,num):
    return random.sample(nodes_list,num)

def save_city_to_file(city):
    city.edges.to_csv('data/random_edges.csv', index=False)
    city.building_nodes.to_csv('data/building_nodes.csv', index=False)
    city.nodes.to_csv('data/random_nodes.csv', index=False)


def load_city():
    edgelist = pd.read_csv('data/random_edges.csv')
    nodelist = pd.read_csv('data/random_nodes.csv')
    buildingnode = pd.read_csv('data/building_nodes.csv')

    g = nx.Graph()

    for i, element in edgelist.iterrows():
        g.add_edge(element[0], element[1], weight=element[2], distance=element[3])

    for i, element in nodelist.iterrows():
        g.nodes[element['id']].update(element[1:].to_dict())

    building_nodes = []
    for i, element in buildingnode.iterrows():
        g.nodes[element['id']].update(element[1:].to_dict())
        dic = {'id': element['id'], 'x': element['x'], 'y': element['y'],'num_bikes': element['num_bikes']}
        building_nodes.append(dic)

    city = City()
    city.graph = g
    city.building_nodes = building_nodes
    city.node_positions = {node[0]: (node[1]['x'], node[1]['y']) for node in city.graph.nodes(data=True)}

    return city


def visualize_city(city):
    fig = plt.figure(dpi=200, figsize=[20, 20])
    ax = fig.add_subplot(111)
    draw_city_on_axes(city, ax)
    plt.axis('square')
    plt.show()


def draw_city_on_axes(city,ax):
    number_id = 0
    for build in city.building_nodes:
        number_id = number_id + 1
        draw_x = [build['x'], build['x']]
        draw_y = [build['y'], build['y'] - 10]
        ax.add_patch(plt.Rectangle((build['x'] - 15, build['y'] - 55), 35, 35, edgecolor='black', facecolor='none'))
        plt.plot(draw_x, draw_y, color='blue', linewidth=1, alpha=0.2)

        if build['num_bikes']>=30:
            plt.text(build['x'], build['y'] - 40, chr(ord('@') + number_id), horizontalalignment='center',
                     verticalalignment='center', fontsize=20, color='darkviolet',weight='bold')
        else:
            plt.text(build['x'], build['y'] - 40, chr(ord('@') + number_id), horizontalalignment='center', verticalalignment='center', fontsize=15, color='mediumorchid')
    #node_positions = {node[0]: (node[1]['x'], node[1]['y']) for node in city.graph.nodes(data=True)}
    label_positions = {node[0]: (node[1]['x'] - 15, node[1]['y'] + 10) for node in city.graph.nodes(data=True)}
    node_labels = {}
    for node in city.graph.nodes(data=True):
        node_labels.update({node[0]: node[0]})
    nx.draw_networkx_nodes(city.graph, pos=city.node_positions, node_size=20, node_color='black', alpha=0.4, ax=ax)
    nx.draw_networkx_labels(city.graph, pos=label_positions, labels=node_labels, font_size=10, font_color='black', horizontalalignment='center', verticalalignment='center', ax=ax)
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edge_color='blue', alpha=0.6, ax=ax)
    bbox = {'ec':[1,1,1,0], 'fc':[1,1,1,0]}
    # hack to label edges over line (rather than breaking up line)
    edge_labels = nx.get_edge_attributes(city.graph,'weight')
    nx.draw_networkx_edge_labels(city.graph, pos=city.node_positions, edge_labels=edge_labels, font_size=10)


def visualize_path_in_city(city, path):
    fig = plt.figure(dpi=200, figsize=[20, 20])
    ax = fig.add_subplot(111)
    draw_city_on_axes(city, ax)
    shortest_path_list = []
    for i in range(len(path) - 1):
        shortest_path_list.append([path[i], path[i + 1]])
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edgelist=shortest_path_list, edge_color='limegreen', ax=ax, width=5)
    plt.axis('square')
    plt.show()

def visualize_rain_path(city, paths,degree):
    fig = plt.figure(dpi=200, figsize=[20, 20])
    ax = fig.add_subplot(111)
    draw_city_on_axes(city, ax)
    if degree=='heavy':
        color='dodgerblue'
    elif degree=='middle':
        color='deepskyblue'
    else:
        color='lightblue'
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edgelist=paths, edge_color=color, ax=ax,
                           width=10,alpha=0.5)
    plt.axis('square')
    plt.show()

def visualize_busy_path(city, paths,degree):
    fig = plt.figure(dpi=200, figsize=[20, 20])
    ax = fig.add_subplot(111)
    draw_city_on_axes(city, ax)
    if degree=='heavy':
        color='red'
    else:
        color='tomato'
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edgelist=paths, edge_color=color, ax=ax,
                           width=10, alpha=0.5)
    plt.axis('square')
    plt.show()

def visualize_both_condition(city, rain_path, traffic_path, rain_degree, traffic_degree):
    fig = plt.figure(dpi=200, figsize=[20, 20])
    ax = fig.add_subplot(111)
    draw_city_on_axes(city, ax)
    if rain_degree == 'heavy':
        color = 'dodgerblue'
    elif rain_degree == 'middle':
        color = 'deepskyblue'
    else:
        color = 'lightblue'
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edgelist=rain_path, edge_color=color, ax=ax,
                           width=10, alpha=0.5)

    if traffic_degree== 'heavy':
        color1 = 'red'
    else:
        color1= 'tomato'
    nx.draw_networkx_edges(city.graph, pos=city.node_positions, edgelist=traffic_path, edge_color=color1, ax=ax,
                           width=10, alpha=0.5)
    plt.axis('square')
    plt.show()


if __name__ == "__main__":
    #my_city = generate_city()
    #save_city_to_file(my_city)
    my_city = load_city()
    visualize_city(my_city)
    p = nx.shortest_path(my_city.graph,'A','B')
    visualize_path_in_city(my_city,p)
