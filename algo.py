import heapq

import pandas as pd

import math

df = pd.read_csv('flights_cut_20240418.csv')
# new_df = df.loc[1:50, ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE']].dropna(axis=0)
new_df = df.loc[1:50, ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ELAPSED_TIME']].dropna(axis=0)
# new_df = df.loc[1:50, ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ELAPSED_TIME']].dropna(axis=0)
unique = list(set().union(new_df.ORIGIN_AIRPORT.unique().tolist(), new_df.DESTINATION_AIRPORT.unique().tolist()))
dist = {}
visited = {}
parent = {}
heap = []


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.adj_list = {}
        for node in self.nodes:
            self.adj_list[node] = []
            dist[node] = math.inf
            visited[node] = False
            parent[node] = -1

    def add_adj(self, u, v, w):
        # lst1 = [u, w]
        temp = (w, v)
        self.adj_list[u].append(temp)
        # self.adj_list[v].append(lst1)

    def print_adj_list(self):
        for node in self.nodes:
            print(node, '->', self.adj_list[node])



def init_sssp(s):
    dist[s] = 0


def dijkstra(g,source):
    init_sssp(source)
    heapq.heappush(heap,(0,source)) # insert sort
    while heap:  # heap not empty
        weight, origin_airport = heapq.heappop(heap)
        visited[origin_airport] = True
        for edge in g[origin_airport]:
            edge_name = edge[1]
            edge_w = edge[0]
            if visited[edge_name]:continue
            if dist[origin_airport] + edge_w < dist[edge_name]:  # tense
                # relaxing
                dist[edge_name] = dist[origin_airport] + edge_w
                heapq.heappush(heap,(dist[edge_name],edge_name))
                parent[edge_name] = origin_airport
    return dist


def shortest_path(orig, dest, arr=None):
    if dist[dest] == math.inf:
        return []
    if arr is None:
        arr = []
    if dest != orig:
        return shortest_path(orig, parent[dest], arr + [dest])
    return arr + [orig]

def total_edge():
    total_edges = []
    for index in range(len(new_df)):
        orig = new_df.iloc[index].ORIGIN_AIRPORT
        dest = new_df.iloc[index].DESTINATION_AIRPORT
        # w = new_df.iloc[index].DISTANCE
        w = new_df.iloc[index].ELAPSED_TIME
        total_edges.append((orig, dest, w))
    print('total edge: ', total_edges)


def set_up_user(class_graph: Graph, graph:dict):
    for index in range(len(new_df)):
        orig = new_df.iloc[index].ORIGIN_AIRPORT
        dest = new_df.iloc[index].DESTINATION_AIRPORT
        w = new_df.iloc[index].ELAPSED_TIME
        # w = new_df.iloc[index].DISTANCE
        my_graph.add_adj(orig, dest, w)

def view_ad(class_graph: Graph):
    pick = input('Do you want to view adjacency list? Yes(1)/No(2) ')
    if pick == '1':
        print('---Adjacency list---')
        print()
        print(class_graph.print_adj_list())
        print()

def prompt(class_graph: Graph):
    orig = None
    dest = None
    while orig not in unique or dest not in unique:
        orig = input('What is your origin airport? ')
        dest = input('What is your destination airport? ')
    init_sssp(orig)
    dijkstra(graph, orig)
    distance = dist[dest]
    arr = shortest_path(orig, dest)
    ans = arr[::-1]
    print()
    print(f'The shortest path from {orig} to {dest} takes {distance} minutes via flight route {ans}')



if __name__ == '__main__':
    my_graph = Graph(unique)
    graph = my_graph.adj_list

    set_up_user(my_graph,graph)
    view_ad(my_graph)
    prompt(my_graph)


    # dijkstra(graph, 'DFW')


    # shortest_path = shortest_path('DFW','PBI')
    # print(shortest_path[::-1])

    # print(total_edge())