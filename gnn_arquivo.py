import zipfile
import tempfile
import networkx as nx
from fastkml import kml
from shapely.geometry import Point, MultiPoint, LineString
from pathlib import Path
from simplekml import Kml
from lxml import etree
import concurrent.futures

# Caminho de entrada e saída
INPUT_FILE = "KMZ_Geral.kmz"   # pode ser .kml ou .kmz
OUTPUT_FILE = "rede_olt_atualizado.kmz"

def load_kml_from_file(filepath: str) -> bytes:
    """Carrega e retorna o conteúdo XML do KML, seja de arquivo KML ou KMZ."""
    path_obj = Path(filepath)
    if path_obj.suffix.lower() == ".kmz":
        with zipfile.ZipFile(str(path_obj), "r") as z:
            # normalmente o arquivo principal dentro do KMZ chama doc.kml
            for name in z.namelist():
                if name.endswith(".kml"):
                    with z.open(name) as f:
                        return f.read()
        raise ValueError("Nenhum arquivo .kml encontrado dentro do KMZ.")
    elif path_obj.suffix.lower() == ".kml":
        return path_obj.read_bytes()
    else:
        raise ValueError("Formato de arquivo não suportado (use .kml ou .kmz).")

# --- 1. Ler arquivo ---
xml_data = load_kml_from_file(INPUT_FILE)
k = kml.KML()
k.from_string(xml_data)

print("[DEBUG] Primeiros 500 caracteres do XML carregado:")
print(xml_data[:500])

# Função para obter placemarks contendo Point, MultiPoint ou LineString usando lxml
def get_all_placemarks_from_kml(xml_bytes):
    nsmap = {
        'kml': 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2'
    }
    root = etree.fromstring(xml_bytes)
    placemarks_data = []

    def get_folder_path(element):
        path = []
        parent = element.getparent()
        while parent is not None:
            if parent.tag.endswith('Folder') or parent.tag.endswith('Document'):
                name_elem = parent.find('kml:name', namespaces=nsmap)
                if name_elem is not None and name_elem.text:
                    path.insert(0, name_elem.text)
            parent = parent.getparent()
        return path

    for pm in root.xpath('.//kml:Placemark', namespaces=nsmap):
        name_elem = pm.find('kml:name', namespaces=nsmap)
        name = name_elem.text if name_elem is not None else ""
        desc_elem = pm.find('kml:description', namespaces=nsmap)
        description = desc_elem.text if desc_elem is not None else ""
        folder_path = get_folder_path(pm)

        geom_type = None
        coords_text = None

        # Check for Point
        point = pm.find('.//kml:Point', namespaces=nsmap)
        if point is not None:
            geom_type = "Point"
            coords_elem = point.find('kml:coordinates', namespaces=nsmap)
            if coords_elem is not None and coords_elem.text:
                coords_text = coords_elem.text.strip()
        else:
            multipoint = pm.find('.//kml:MultiGeometry', namespaces=nsmap)
            if multipoint is not None:
                # Check if MultiGeometry contains multiple Points
                points = multipoint.findall('kml:Point', namespaces=nsmap)
                if points and len(points) > 1:
                    geom_type = "MultiPoint"
                    coords_list = []
                    for pt in points:
                        coords_elem = pt.find('kml:coordinates', namespaces=nsmap)
                        if coords_elem is not None and coords_elem.text:
                            coords_list.append(coords_elem.text.strip())
                    coords_text = ";".join(coords_list)
                else:
                    # Could be other geometries inside MultiGeometry, check for LineString
                    linestring = multipoint.find('kml:LineString', namespaces=nsmap)
                    if linestring is not None:
                        geom_type = "LineString"
                        coords_elem = linestring.find('kml:coordinates', namespaces=nsmap)
                        if coords_elem is not None and coords_elem.text:
                            coords_text = coords_elem.text.strip()
            else:
                # Check for LineString directly
                linestring = pm.find('.//kml:LineString', namespaces=nsmap)
                if linestring is not None:
                    geom_type = "LineString"
                    coords_elem = linestring.find('kml:coordinates', namespaces=nsmap)
                    if coords_elem is not None and coords_elem.text:
                        coords_text = coords_elem.text.strip()

        if geom_type and coords_text:
            placemarks_data.append((name, geom_type, coords_text, folder_path, description))

    return placemarks_data

placemarks = get_all_placemarks_from_kml(xml_data)

# --- 1.5. Gerar arquivo intermediário CSV com placemarks capturados ---
import csv

CSV_INTERMEDIARIO = "placemarks_capturados.csv"
with open(CSV_INTERMEDIARIO, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["nome_do_placemark", "tipo_geometria", "coordenadas", "folder_path", "description"])
    for nome, tipo, coords_str, folder_path, descricao in placemarks:
        folder_str = " / ".join([f for f in folder_path if f])
        writer.writerow([nome, tipo, coords_str, folder_str, descricao])

# --- 2. Criar grafo ---
G = nx.Graph()

# Função auxiliar para criar ou obter um node_id para um ponto
def get_node_id_for_point(point, base_name="pt"):
    # Usar coordenadas como parte do node_id para garantir unicidade
    return f"{base_name}_{point.x:.6f}_{point.y:.6f}"

# Guardar mapa de node para folder path
node_to_folder = {}

for nome, tipo, coords_str, folder_path, descricao in placemarks:
    if tipo == "Point":
        parts = coords_str.split(',')
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            coords = (x, y)
            node_id = nome if nome else get_node_id_for_point(Point(x, y), "node")
            G.add_node(node_id, coords=coords, description=descricao)
            node_to_folder[node_id] = folder_path
    elif tipo == "MultiPoint":
        points_str = coords_str.split(';')
        for idx, pt_str in enumerate(points_str):
            parts = pt_str.split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords = (x, y)
                node_id = f"{nome}_{idx}" if nome else get_node_id_for_point(Point(x, y), f"multipoint_{idx}")
                G.add_node(node_id, coords=coords, description=descricao)
                node_to_folder[node_id] = folder_path
    elif tipo == "LineString":
        coords_list = []
        coord_pairs = coords_str.split()
        for c in coord_pairs:
            parts = c.split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords_list.append((x, y))
        node_ids = []
        for idx, coord in enumerate(coords_list):
            if nome:
                node_id = f"{nome}_{idx}"
            else:
                node_id = get_node_id_for_point(Point(coord), f"linestring_{idx}")
            if node_id not in G:
                G.add_node(node_id, coords=coord, description=descricao)
                node_to_folder[node_id] = folder_path
            node_ids.append(node_id)
        for i in range(len(node_ids) - 1):
            G.add_edge(node_ids[i], node_ids[i+1], fibra="SMF")

# Ensure "OLT1" node exists in the graph
if "OLT1" not in G:
    G.add_node("OLT1", coords=(0.0, 0.0), description="OLT padrão")
    node_to_folder["OLT1"] = ["OLT1"]

# Exemplo de ligação: ONU -> OLT1
for n in list(G.nodes()):
    if n.startswith("ONU"):
        G.add_edge("OLT1", n, fibra="SMF")

# --- 3. Atualizar atributos (simulação de coleta SNMP) ---

def update_node_status(n):
    if n.startswith("ONU"):
        G.nodes[n]["status"] = "ativo"
    if n.startswith("OLT"):
        G.nodes[n]["status"] = "ativo"

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(update_node_status, n) for n in G.nodes()]
    for future in concurrent.futures.as_completed(futures):
        future.result()

# --- 4. Exportar novo KML preservando hierarquia de pastas ---
kml_out = Kml()

# Função para obter ou criar uma Folder no simplekml dado o caminho da hierarquia
def get_or_create_folder(kml_parent, folder_path):
    current = kml_parent
    for folder_name in folder_path:
        # Procurar folder existente com esse nome
        found = None
        for f in current.features:
            if isinstance(f, type(kml_out.newfolder())) and f.name == folder_name:
                found = f
                break
        if found is None:
            found = current.newfolder(name=folder_name)
        current = found
    return current

# Criar folders e adicionar pontos
# Primeiro criar um dict para guardar folders criados para paths
folder_cache = {}

for n, data in G.nodes(data=True):
    folder_path = node_to_folder.get(n, [])
    folder_key = tuple(folder_path)
    if folder_key not in folder_cache:
        folder_cache[folder_key] = get_or_create_folder(kml_out, folder_path)
    folder = folder_cache[folder_key]
    x, y = data["coords"]
    pnt = folder.newpoint(name=n, coords=[(x, y)])
    desc = f"Status: {data.get('status','desconhecido')}"
    pnt.description = desc

# Adicionar arestas (linhas) no mesmo folder dos seus nós iniciais (se ambos tiverem a mesma pasta)
for u, v, attr in G.edges(data=True):
    folder_u = node_to_folder.get(u, [])
    folder_v = node_to_folder.get(v, [])
    if folder_u == folder_v:
        folder_key = tuple(folder_u)
        if folder_key not in folder_cache:
            folder_cache[folder_key] = get_or_create_folder(kml_out, folder_u)
        folder = folder_cache[folder_key]
        coords = [G.nodes[u]['coords'], G.nodes[v]['coords']]
        line = folder.newlinestring(name=f"{u} to {v}", coords=coords)
        line.description = attr.get("fibra", "")
        line.style.linestyle.width = 2
        line.style.linestyle.color = "ff0000ff"  # Red color in aabbggrr hex

# Salvar como KML temporário
with tempfile.TemporaryDirectory() as tmpdir:
    kml_path = Path(tmpdir) / "doc.kml"
    kml_out.save(str(kml_path))
    
    # Compactar em KMZ
    with zipfile.ZipFile(OUTPUT_FILE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(kml_path, "doc.kml")

print(f"Arquivo atualizado salvo em {OUTPUT_FILE}")