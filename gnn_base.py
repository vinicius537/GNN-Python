# pip install torch torch-geometric simplekml networkx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import simplekml

# --------------------------
# 1. Criando grafo fictício com OLT -> Fibras -> ONUs
# --------------------------
G = nx.Graph()

# OLT
G.add_node(0, tipo="OLT", potencia=-5.0, estado=1)

# Fibras intermediárias (como splitters/caixas)
G.add_node(10, tipo="Fibra", potencia=-10.0, estado=1)
G.add_node(11, tipo="Fibra", potencia=-12.0, estado=1)

# ONUs
G.add_node(1, tipo="ONU", potencia=-20.0, estado=1)   # ok
G.add_node(2, tipo="ONU", potencia=-28.0, estado=0)   # anômala
G.add_node(3, tipo="ONU", potencia=-15.0, estado=1)   # ok
G.add_node(4, tipo="ONU", potencia=-30.0, estado=0)   # anômala

# Conexões físicas
G.add_edges_from([
    (0,10),       # OLT -> Fibra 10
    (0,11),       # OLT -> Fibra 11
    (10,1), (10,2),  # Fibra 10 -> ONU1/ONU2
    (11,3), (11,4)   # Fibra 11 -> ONU3/ONU4
])

# --------------------------
# 2. Convertendo para PyTorch Geometric
# --------------------------
# Features: [potência normalizada, estado]
x = []
for node in G.nodes:
    potencia = G.nodes[node]["potencia"] / 30.0
    estado = G.nodes[node]["estado"]
    x.append([potencia, estado])

x = torch.tensor(x, dtype=torch.float)

# Mapear IDs para índices
id_map = {nid: i for i, nid in enumerate(G.nodes())}

# Arestas
edge_index = []
for u,v in G.edges:
    edge_index.append([id_map[u], id_map[v]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Labels (0 = ok, 1 = anômala)
y = []
for node in G.nodes:
    if G.nodes[node]["tipo"] == "ONU":
        y.append(1 if G.nodes[node]["estado"] == 0 else 0)
    else:
        y.append(0)  # OLT/Fibra marcados como ok
y = torch.tensor(y, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# --------------------------
# 3. Definindo uma GNN simples
# --------------------------
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --------------------------
# 4. Treinamento
# --------------------------
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y) # type: ignore
    loss.backward()
    optimizer.step()

pred = out.argmax(dim=1)

print("Labels reais:", data.y.tolist()) # type: ignore
print("Predições  :", pred.tolist())

# --------------------------
# 5. Exportando para KML
# --------------------------
coords = {
    0: (-38.50, -3.70),   # OLT
    10: (-38.51, -3.701), # Fibra 10
    11: (-38.50, -3.702), # Fibra 11
    1: (-38.52, -3.703),  # ONU1
    2: (-38.53, -3.704),  # ONU2
    3: (-38.49, -3.705),  # ONU3
    4: (-38.48, -3.706),  # ONU4
}

kml = simplekml.Kml()

# Adicionar nós
for node in G.nodes:
    nome = G.nodes[node]["tipo"] + f" {node}"
    estado = "Anômala" if pred[id_map[node]].item() == 1 else "OK"
    p = kml.newpoint(
        name=f"{nome} ({estado})",
        coords=[coords[node]]
    )
    if G.nodes[node]["tipo"] == "OLT":
        p.style.iconstyle.color = simplekml.Color.blue
    elif G.nodes[node]["tipo"] == "Fibra":
        p.style.iconstyle.color = simplekml.Color.yellow
    else:  # ONU
        if estado == "Anômala":
            p.style.iconstyle.color = simplekml.Color.red
        else:
            p.style.iconstyle.color = simplekml.Color.green

# Adicionar arestas como linhas (fibras físicas)
for u,v in G.edges:
    line = kml.newlinestring(
        name=f"{u}-{v}",
        coords=[coords[u], coords[v]]
    )
    line.style.linestyle.color = simplekml.Color.white
    line.style.linestyle.width = 2

kml.save("rede_ftth.kml")
print("Arquivo KML gerado: rede_ftth.kml")