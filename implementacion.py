import torch

# Importamos numpy
import numpy as np

# importamos para la visualizacion
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.datasets import KarateClub

# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Numero de grafos: {len(dataset)}')
print(f'Numero de caracteristicas: {dataset.num_features}')
print(f'Numero de clases: {dataset.num_classes}')

#pintamos el primer elemento
print(f'Grafo: {dataset[0]}')

#el x= [34, 34] es la matriz de caracteristicas del nodo con forma (num nodos, num caracteristicas).add()
#el y= [34] es el nodo Ground Truth Labels. tenemos un valor para cada nodo
#el edge_index= [2, 156] representa la conectividad del grafo (como se conectan los nodos) con forma (2, num aristas)
#el train_mask= [34] es una mascara binaria que indica si el nodo es de entrenamiento o no. es opcional

#imprimimos cada uno de estos tensores:
#características
data = dataset[0]
print(f'x = {data.x.shape}')
print(data.x)
'''vemos que esta matriz es una matriz de identidad. No nos proporciona ninguna información sobre los nodos,
así que los tendremos que clasificar mirando sus conexiones.'''

#imprimimos el indice de borde:
print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)

'''tenemos dos listas de 156 aristas dirigidas (78 aristas bidireccionales). la primera lista contiene 
las fuentes y la segunda los destinos. esto se llama lista de coordenadas (COO)'''

'''Otra manera de representar los grafos es mediante una matriz de adyacencia simple, donde un elemento distinto de 0 indica una conexion.'''
#matriz de adyacencia
from torch_geometric.utils import to_dense_adj
A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A = {A.shape}')
print(A)

#imprimimos las etiquetas:
print(f'y = {data.y.shape}')
print(data.y)
'''Codifican el numero de grupo (0, 1, 2, 3) para cada nodo, por lo que tenemos 34 valores'''

#imprimimos la mascara del tren:
print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)
'''Esta mascara binaria indica si el nodo es de entrenamiento o no. En este caso, todos los nodos son de entrenamiento.
Estos deben usarse para entrenar con declaraciones True,  mientras que los demas pueden considerarse como el conjunto de prueba.'''

#Vemos otras propiedades del grafo
print(f'Los bordes están dirigidos: {data.is_directed()}') #dice si el grafo está dirigido. Es decir, que la matriz de adyacencia no es simétrica
print(f'El grafo tiene nodos aislados: {data.has_isolated_nodes()}') #Comprueba si hay nodos que no están conectados con el resto del grafo
print(f'El grafo tiene lazos: {data.has_self_loops()}') #Comprueba si hay nodos que se conectan a sí mismos

'''Vamos a visualizar el grafo con NetworkX y Matplotlib. Para ello, vamos a convertir el grado de PyTorch Geometriz a la biblioteca de NetworkX'''
#Trazamos el conjunto de datos con un color diferente para cada grupo