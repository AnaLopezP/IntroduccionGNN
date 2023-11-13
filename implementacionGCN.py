'''Creamos un GCN simple con una sola capa de GCN, una fujcion de activacion de ReLU y una capa lineal.
Esta capa final generará 4 valores, correspondientes a las 4 clases de nodos. El valor más alto determinará la clase del nodo.'''

'''Definimos la capa GCN con una capa oculta de 3 dimensiones.'''
import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub

import IPython
from IPython.display import HTML
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

matplotlib.use('Agg')
dataset = KarateClub()
data = dataset[0]
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = torch.nn.Linear(3, dataset.num_classes)
        
    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        model = GCN()
        print(model)
        return h, z, model

model = GCN()
print(model)

'''Escribimos un ciclo de entrenamiento simple con PyTorch. Tratamos de predecir las etiquetas correctas y comparamos los resultados del GCN con los valores almacenados en data.y
el error se calcula mediante la perdida de entropia cruzada y se retropropaga con Adam para afinar los pesos y sesgos de la GNN.'''

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

#calculamos la exactitud
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

#data para animaciones
embeddings = []
losses = []
accuracies = []
outputs = []

# Bucle de entrenamiento
for epoch in range(201):
    # Reiniciar gradientes
    optimizer.zero_grad()

    # Pase hacia adelante (Forward pass)
    result = model(data.x, data.edge_index)
    h, z = result[0], result[1]

    # Calcular la función de pérdida
    loss = criterion(z, data.y)

    # Calcular la precisión
    acc = accuracy(z.argmax(dim=1), data.y)

    # Calcular gradientes
    loss.backward()

    # Ajustar parámetros
    optimizer.step()

    # Almacenar datos para animaciones
    embeddings.append(h)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # Imprimir métricas cada 10 épocas
    if epoch % 10 == 0:
        print(f'Época {epoch:>3} | Pérdida: {loss:.2f} | Precisión: {acc*100:.2f}%')


'''Vemos que alcanzamos el 100% de precision en el conunto de entrenamiento
podemos producir una visualizacion clara animando el gráfico
y ver la evolucion de las predicciones de la GNN durante el proceso de entrenamiento'''

plt.rcParams["animation.bitrate"] = 3000
def animate(i):
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color=outputs[i],
                    cmap="hsv",
                    vmin=-2,
                    vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14
                    )
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=20)
    
fig = plt.figure(figsize=(12, 12))
plt.axis('off')
anim = animation.FuncAnimation(fig, animate, \
            np.arange(0, 200, 10), interval=500, repeat=True)
animatione_html = HTML(anim.to_jshtml())

IPython.display.display(animatione_html)

#Printeamos las incrustaciones aprendidas
print(f'Final embeddings = {h.shape}')
print(h)



#Obtener la primera incrustación en la época = 0
embed = h[0].detach().cpu().numpy()

fig = plt.figure(figsize=(12, 12))
plt.axis('off')
ax = fig.add_subplot(projection='3d')
ax.patch.set_alpha(0)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

data_y_flat = data.y.numpy().ravel()
embed = embed.reshape(-1, 3)
ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data_y_flat, cmap="hsv", vmin=-2, vmax=3)

plt.show()



#Veamos cómo evolucionan con el tiempo, a medida que GCN mejora cada vez más en la clasificación de nodos.

def animate(i):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    color_values = outputs[i].detach().cpu().numpy().ravel()
    scatter = ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=color_values, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)
    #añadimos una barra de colores
    plt.colorbar(scatter)

fig = plt.figure(figsize=(12, 12))
plt.axis('off')
ax = fig.add_subplot(projection='3d')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

anim = animation.FuncAnimation(fig, animate, \
              np.arange(0, 200, 10), interval=800, repeat=True)
html = HTML(anim.to_html5_video())
