# pip install torch torch-geometric simplekml networkx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import simplekml

# --------------------------
# 1. Criando grafo com 2 OLTs -> fibras -> ONUs
# --------------------------
G = nx.Graph()

# OLTs
G.add_node(0, tipo="OLT", potencia=-5.0, estado=1, grupo="OLT_A")
G.add_node(100, tipo="OLT", potencia=-6.0, estado=1, grupo="OLT_B")

# Fibras OLT_A
G.add_node(10, tipo="Fibra", potencia=-10.0, estado=1, grupo="OLT_A")
G.add_node(11, tipo="Fibra", potencia=-12.0, estado=1, grupo="OLT_A")

# ONUs OLT_A
G.add_node(1, tipo="ONU", potencia=-20.0, estado=1, grupo="OLT_A")
G.add_node(2, tipo="ONU", potencia=-28.0, estado=0, grupo="OLT_A")

# Fibras OLT_B
G.add_node(110, tipo="Fibra", potencia=-11.0, estado=1, grupo="OLT_B")

# ONUs OLT_B
G.add_node(101, tipo="ONU", potencia=-18.0, estado=1, grupo="OLT_B")
G.add_node(102, tipo="ONU", potencia=-27.0, estado=0, grupo="OLT_B")
G.add_node(103, tipo="ONU", potencia=-25.0, estado=1, grupo="OLT_B")

# Conexões físicas
G.add_edges_from([
    (0,10), (0,11), (10,1), (11,2),          # OLT_A
    (100,110), (110,101), (110,102), (110,103) # OLT_B
])

# --------------------------
# 2. Convertendo para PyTorch Geometric
# --------------------------
x = []
for node in G.nodes:
    potencia = G.nodes[node]["potencia"] / 30.0
    estado = G.nodes[node]["estado"]
    x.append([potencia, estado])

x = torch.tensor(x, dtype=torch.float)
id_map = {nid: i for i, nid in enumerate(G.nodes())}

edge_index = []
for u,v in G.edges:
    edge_index.append([id_map[u], id_map[v]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

y = []
for node in G.nodes:
    if G.nodes[node]["tipo"] == "ONU":
        y.append(1 if G.nodes[node]["estado"] == 0 else 0)
    else:
        y.append(0)
y = torch.tensor(y, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# --------------------------
# 3. GNN
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
# 4. Coordenadas fictícias
# --------------------------
coords = {
    # OLT_A
    0: (-38.50, -3.70),
    10: (-38.51, -3.701),
    11: (-38.52, -3.702),
    1: (-38.53, -3.703),
    2: (-38.54, -3.704),
    # OLT_B
    100: (-38.60, -3.72),
    110: (-38.61, -3.721),
    101: (-38.62, -3.722),
    102: (-38.63, -3.723),
    103: (-38.64, -3.724),
}

# --------------------------
# 5. Exportando para KML (com folders por OLT)
# --------------------------
kml = simplekml.Kml()

folders = {}
for node in G.nodes:
    grupo = G.nodes[node]["grupo"]
    if grupo not in folders:
        folders[grupo] = kml.newfolder(name=grupo)

# Adicionar nós
for node in G.nodes:
    nome = G.nodes[node]["tipo"] + f" {node}"
    estado = "Anômala" if pred[id_map[node]].item() == 1 else "OK"
    grupo = G.nodes[node]["grupo"]

    p = folders[grupo].newpoint(
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

# Adicionar conexões
for u,v in G.edges:
    grupo = G.nodes[u]["grupo"]
    line = folders[grupo].newlinestring(
        name=f"{u}-{v}",
        coords=[coords[u], coords[v]]
    )
    line.style.linestyle.color = simplekml.Color.white
    line.style.linestyle.width = 2

kml.save("redes_multiplas_OLTs.kml")
print("Arquivo KML gerado: redes_multiplas_OLTs.kml")