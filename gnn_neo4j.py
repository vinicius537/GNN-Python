import argparse
import asyncio
import concurrent.futures
import gc
import logging
import os
import sys
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Generator

import psutil
from lxml import etree
import networkx as nx
from fastkml import kml
from shapely.geometry import Point
from memory_profiler import profile
from neo4j import AsyncGraphDatabase

# PyTorch e PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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
import networkx as nx
import asyncio
from neo4j import AsyncGraphDatabase
import logging
from typing import Dict, List, Any, Optional, Tuple, Generator
import concurrent.futures
import zipfile
from pathlib import Path
from fastkml import kml
from shapely.geometry import Point
import argparse
import os
import sys
import gc
import psutil
from datetime import datetime
import xml.etree.ElementTree as ET
from functools import partial
from memory_profiler import profile


def process_kml_chunk(chunk: bytes):
    """
    Processa um chunk de KML extraindo ONUs (Placemarks com Point).
    Retorna uma lista de dicionários: {id, coords, type, description}
    """
    results = []
    try:
        # Parse o XML do chunk
        root = ET.fromstring(chunk)
        # Namespaces comuns em KML
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        # Busca todos os Placemarks
        for placemark in root.findall('.//kml:Placemark', ns):
            # Tenta extrair o nome/id
            name_elem = placemark.find('kml:name', ns)
            node_id = name_elem.text.strip() if name_elem is not None else None
            # Tenta extrair a descrição
            desc_elem = placemark.find('kml:description', ns)
            description = desc_elem.text.strip() if desc_elem is not None else ''
            # Tenta extrair coordenadas de Point
            point_elem = placemark.find('.//kml:Point', ns)
            coords = None
            if point_elem is not None:
                coord_elem = point_elem.find('kml:coordinates', ns)
                if coord_elem is not None and coord_elem.text:
                    coord_text = coord_elem.text.strip()
                    # KML: lon,lat[,alt]
                    parts = coord_text.split(',')
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            coords = (lat, lon)
                        except Exception:
                            coords = None
            # Só adiciona se tiver id e coordenadas
            if node_id and coords:
                results.append({
                    'id': node_id,
                    'coords': coords,
                    'type': 'ONU',
                    'description': description
                })
    except Exception as e:
        # Loga erro, mas não interrompe processamento dos outros chunks
        import logging
        logging.getLogger(__name__).warning(f"Erro ao processar chunk de KML: {e}")
    return results

def setup_logging():
    """Configure optimized logging with file and console output."""
    log_filename = f"gnn_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    handlers = [
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# Initialize optimized logging
logger = setup_logging()


class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.process = psutil.Process()
        self.start_time = datetime.now()
        
    def log_usage(self):
        """Log current resource usage."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            logger.info(
                f"Resource Usage - "
                f"Memory: {memory_info.rss / 1024 / 1024:.2f} MB, "
                f"CPU: {cpu_percent}%, "
                f"Runtime: {datetime.now() - self.start_time}"
            )
        except Exception as e:
            logger.warning(f"Failed to log resource usage: {str(e)}")
            
    def clear_memory(self):
        """Force garbage collection to free memory."""
        try:
            gc.collect()
            logger.debug("Performed memory cleanup")
        except Exception as e:
            logger.warning(f"Failed to clear memory: {str(e)}")

def validate_file(filepath: str) -> bool:
    """
    Validate if the file exists and has a supported extension.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    supported_extensions = ['.kml', '.kmz']
    path = Path(filepath)
    
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        return False
        
    if path.suffix.lower() not in supported_extensions:
        logger.error(f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}")
        return False
    
    # Check file size
    file_size = path.stat().st_size
    if file_size == 0:
        logger.error("File is empty")
        return False
    
    logger.info(f"Validated file: {filepath} (Size: {file_size / 1024 / 1024:.2f} MB)")
    return True

def split_xml_content(xml_bytes: bytes, chunk_size: int = 1024*1024) -> List[bytes]:
    """
    Split large XML content into manageable chunks for parallel processing.
    
    Args:
        xml_bytes: The XML content as bytes
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List[bytes]: List of XML chunks
    """
    chunks = []
    total_size = len(xml_bytes)
    
    for i in range(0, total_size, chunk_size):
        chunk = xml_bytes[i:i + chunk_size]
        # Ensure we don't split in the middle of an XML element
        if i + chunk_size < total_size:
            # Find the last complete element
            last_end = chunk.rfind(b'</') 
            if last_end != -1:
                next_start = chunk.find(b'>', last_end)
                if next_start != -1:
                    chunk = chunk[:next_start + 1]
                    
        chunks.append(chunk)
    
    return chunks

@profile
def load_kml_content(filepath: str) -> bytes:
    """
    Load KML content from either KML or KMZ file with optimized memory usage.
    
    Args:
        filepath: Path to the KML or KMZ file
        
    Returns:
        bytes: The KML content as bytes
    """
    logger.info(f"Loading file: {filepath}")
    path = Path(filepath)
    monitor = ResourceMonitor()
    
    try:
        if path.suffix.lower() == '.kmz':
            with zipfile.ZipFile(str(path), 'r') as z:
                for name in z.namelist():
                    if name.endswith('.kml'):
                        logger.info(f"Found KML in KMZ: {name}")
                        with z.open(name) as kml_file:
                            # Read in chunks to handle large files
                            content = bytearray()
                            chunk_size = 1024 * 1024  # 1MB chunks
                            while True:
                                chunk = kml_file.read(chunk_size)
                                if not chunk:
                                    break
                                content.extend(chunk)
                                monitor.log_usage()
                            return bytes(content)
                raise ValueError("No .kml file found in KMZ")
        else:
            # Read KML file in chunks
            with open(path, 'rb') as f:
                content = bytearray()
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content.extend(chunk)
                    monitor.log_usage()
                return bytes(content)
                    
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise
    finally:
        monitor.clear_memory()

class NetworkBuilder:
    def __init__(self):
        self.G = nx.Graph()
        
    def create_olt(self, olt_id: str = "OLT1", coords: tuple = (0.0, 0.0), description: str = "OLT padrão"):
        """Create an OLT node in the graph."""
        self.G.add_node(olt_id, coords=coords, description=description, type="OLT")
        logger.info(f"Created OLT node: {olt_id}")
        return olt_id

    def create_onu(self, onu_id: str, coords: tuple, description: str = "", connect_to_olt: str = "OLT1"):
        """Create an ONU node and optionally connect it to an OLT."""
        self.G.add_node(onu_id, coords=coords, description=description, type="ONU")
        if connect_to_olt:
            self.G.add_edge(onu_id, connect_to_olt, fibra="SMF")
        return onu_id

    def create_fiber_connection(self, source_id: str, target_id: str, fiber_type: str = "SMF"):
        """Create a fiber connection between two nodes."""
        self.G.add_edge(source_id, target_id, fibra=fiber_type)

    def update_node_status(self, node_id: str, status: str):
        """Update the status of a node."""
        if node_id in self.G.nodes:
            self.G.nodes[node_id]["status"] = status

    def bulk_update_status(self):
        """Update status for all nodes in parallel."""
        def update_status(n):
            if self.G.nodes[n].get("type") in ["ONU", "OLT"]:
                self.G.nodes[n]["status"] = "ativo"

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_status, n) for n in self.G.nodes()]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def get_graph(self) -> nx.Graph:
        """Return the NetworkX graph."""
        return self.G

class AsyncNeo4jGraph:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = AsyncGraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=30
        )
        self.batch_size = 10000  # Increased batch size for better performance
        self.monitor = ResourceMonitor()
        
    def stream_data(self, data: List, batch_size: int = None) -> Generator:
        """
        Stream data in batches to manage memory usage.
        
        Args:
            data: List of items to stream
            batch_size: Size of each batch (default: self.batch_size)
            
        Yields:
            List: A batch of items
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        for i in range(0, len(data), batch_size):
            self.monitor.log_usage()  # Log resource usage
            yield data[i:i + batch_size]
            self.monitor.clear_memory()  # Clean up after each batch
        
    async def close(self):
        """Close the Neo4j driver connection."""
        await self.driver.close()
        
    async def clear_database(self):
        """Clear all nodes and relationships in the database."""
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            
    async def create_indexes(self):
        """Create indexes for better query performance."""
        async with self.driver.session() as session:
            try:
                
                # Create index for node ID
                await session.run("CREATE INDEX node_id IF NOT EXISTS FOR (n:Node) ON (n.id)")
                # Create index for node type
                await session.run("CREATE INDEX node_type IF NOT EXISTS FOR (n:Node) ON (n.type)")
                logger.info("Created Neo4j indexes successfully")
            except Exception as e:
                logger.error(f"Failed to create indexes: {str(e)}")
                raise

    async def batch_create_nodes(self, nodes_data: List[Dict[str, Any]]):
        """
        Create nodes in batches using UNWIND with periodic commit for better performance.
        
        Args:
            nodes_data: List of node dictionaries with properties
        """
        query = """
        CALL {
            WITH $batch AS batch
            UNWIND batch AS node
            CREATE (n:Node)
            SET n = node
        } IN TRANSACTIONS OF 1000 ROWS
        """
        
        total_nodes = len(nodes_data)
        logger.info(f"Starting batch creation of {total_nodes} nodes")
        
        async with self.driver.session() as session:
            try:
                # Create indexes first
                await self.create_indexes()
                
                # Process in streams
                for i, batch in enumerate(self.stream_data(nodes_data), 1):
                    await session.run(query, {"batch": batch})
                    logger.info(f"Processed node batch {i} ({len(batch)} nodes)")
                    
                logger.info(f"Successfully created {total_nodes} nodes")
                
            except Exception as e:
                logger.error(f"Error in batch node creation: {str(e)}")
                raise
                    
    async def batch_create_relationships(self, edges_data: List[Dict[str, Any]]):
        """
        Create relationships between nodes in batches using UNWIND with optimized query.
        
        Args:
            edges_data: List of edge dictionaries with source_id, target_id, and properties
        """
        # Optimized query using PERIODIC COMMIT and efficient index usage
        query = """
        CALL {
            WITH $batch AS batch
            UNWIND batch AS edge
            MATCH (source:Node {id: edge.source_id})
            MATCH (target:Node {id: edge.target_id})
            CREATE (source)-[r:CONNECTS_TO]->(target)
            SET r = edge.properties
        } IN TRANSACTIONS OF 1000 ROWS
        """
        
        total_edges = len(edges_data)
        logger.info(f"Starting batch creation of {total_edges} relationships")
        
        async with self.driver.session() as session:
            try:
                # Process in streams
                for i, batch in enumerate(self.stream_data(edges_data), 1):
                    start_time = datetime.now()
                    await session.run(query, {"batch": batch})
                    duration = datetime.now() - start_time
                    
                    logger.info(
                        f"Processed edge batch {i} ({len(batch)} edges) "
                        f"in {duration.total_seconds():.2f} seconds"
                    )
                    
                logger.info(f"Successfully created {total_edges} relationships")
                
            except Exception as e:
                logger.error(f"Error in batch relationship creation: {str(e)}")
                raise

    async def import_networkx_graph(self, G: nx.Graph):
        """
        Import a NetworkX graph into Neo4j using batch operations.
        
        Args:
            G: NetworkX graph instance
        """
        # Prepare nodes data
        nodes_data = []
        for node_id, data in G.nodes(data=True):
            node_data = {"id": str(node_id)}
            node_data.update(data)  # Add all node attributes
            nodes_data.append(node_data)
            
        # Prepare edges data
        edges_data = []
        for u, v, data in G.edges(data=True):
            edge_data = {
                "source_id": str(u),
                "target_id": str(v),
                "properties": data
            }
            edges_data.append(edge_data)
            
        # Clear existing data
        await self.clear_database()
        
        # Create nodes and relationships in batches
        await self.batch_create_nodes(nodes_data)
        await self.batch_create_relationships(edges_data)
        
    async def update_node_status(self, node_id: str, status: str):
        """Update the status of a node."""
        async with self.driver.session() as session:
            query = """
            MATCH (n:Node {id: $node_id})
            SET n.status = $status
            """
            await session.run(query, {"node_id": node_id, "status": status})
            
    async def get_graph_data(self) -> Dict[str, List]:
        """
        Retrieve all nodes and their connections from the database.
        Returns a dictionary with 'nodes' and 'edges' lists.
        """
        async with self.driver.session() as session:
            # Get all nodes
            node_result = await session.run("""
                MATCH (n:Node)
                RETURN collect(n) as nodes
            """)
            nodes = await node_result.single()
            nodes = nodes["nodes"]
            
            # Get all relationships
            edge_result = await session.run("""
                MATCH (source:Node)-[r:CONNECTS_TO]->(target:Node)
                RETURN collect({
                    source_id: source.id,
                    target_id: target.id,
                    properties: properties(r)
                }) as edges
            """)
            edges = await edge_result.single()
            edges = edges["edges"]
            
            return {
                "nodes": nodes,
                "edges": edges
            }

async def export_to_neo4j(G: nx.Graph, uri: str, username: str, password: str):
    """
    Export the NetworkX graph to Neo4j database.
    
    Args:
        G: NetworkX graph instance
        uri: Neo4j database URI
        username: Neo4j username
        password: Neo4j password
    """
    neo4j_graph = AsyncNeo4jGraph(uri, username, password)
    try:
        await neo4j_graph.import_networkx_graph(G)
        logger.info("Graph successfully exported to Neo4j")
    finally:
        await neo4j_graph.close()

# Add Neo4j export functionality to the main flow
async def export_graph_to_neo4j(G: nx.Graph, neo4j_config: Dict[str, str]):
    """Export the graph to Neo4j after processing."""
    try:
        await export_to_neo4j(
            G,
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", "neo4j")
        )
    except Exception as e:
        logger.error(f"Failed to export to Neo4j: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process network topology file and export to Neo4j')

    env_input = os.environ.get('INPUT_FILE')

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (KML ou KMZ). Pode ser definido pela variável de ambiente INPUT_FILE.',
        required=(env_input is None)
    )

    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7687',
        help='Neo4j database URI (default: bolt://localhost:7687)'
    )

    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )

    parser.add_argument(
        '--neo4j-password',
        type=str,
        default='gnnneutra',
        help='Neo4j password (default: neo4j)'
    )

    args = parser.parse_args()
    if getattr(args, 'input', None) is None and env_input is not None:
        args.input = env_input
    return args

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

def nx_to_pyg(G: nx.Graph):
    node_map = {n: i for i, n in enumerate(G.nodes())}
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_map[u], node_map[v]])
    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Features dummy (pode ser ajustado para atributos reais)
    x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(emb_dim * 2, num_classes)
    def forward(self, emb_u, emb_v):
        x = torch.cat([emb_u, emb_v], dim=-1)
        return self.fc(x)

async def main():
    """Main execution function with optimized resource usage."""
    monitor = ResourceMonitor()
    start_time = datetime.now()
    
    try:
        # Parse arguments
        args = parse_arguments()
        # Validate input file
        if not validate_file(args.input):
            return
        monitor.log_usage()

        # --- 1. Ler arquivo ---
        xml_data = load_kml_content(args.input)
        logger.info("Successfully loaded KML content")
        monitor.log_usage()

        # --- 2. Extrair placemarks ---
        placemarks = get_all_placemarks_from_kml(xml_data)

        # --- 3. Criar grafo ---
        G = nx.Graph()
        node_to_folder = {}

        def get_node_id_for_point(point, base_name="pt"):
            return f"{base_name}_{point.x:.6f}_{point.y:.6f}"

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

        # Garantir OLT
        if "OLT" not in G:
            G.add_node("OLT", coords=(0.0, 0.0), description="OLT padrão")
            node_to_folder["OLT"] = ["OLT"]

        # Exemplo de ligação: ONU -> OLT
        for n in list(G.nodes()):
            if n.startswith("ONU"):
                G.add_edge("OLT", n, fibra="SMF")

        # Atualizar status
        def update_node_status(n):
            if n.startswith("ONU"):
                G.nodes[n]["status"] = "ativo"
            if n.startswith("OLT"):
                G.nodes[n]["status"] = "ativo"

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_node_status, n) for n in G.nodes()]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Exportar para Neo4j
        neo4j_config = {
            "uri": args.neo4j_uri,
            "username": args.neo4j_user,
            "password": args.neo4j_password
        }
        await export_graph_to_neo4j(G, neo4j_config)
        logger.info("Successfully exported graph to Neo4j")
        monitor.log_usage()
        duration = datetime.now() - start_time
        logger.info(f"Total processing time: {duration}")
        
        # --- INFERÊNCIA GNN NA GPU ---
        pyg_data = nx_to_pyg(G)
        pyg_data = pyg_data.to('cuda')

        model = SimpleGNN(in_channels=1, hidden_channels=16, out_channels=8)
        # Se você já treinou um modelo, carregue-o aqui:
        # model.load_state_dict(torch.load('meu_modelo_gnn.pth'))
        model = model.to('cuda')
        model.eval()

        with torch.no_grad():
            node_embeddings = model(pyg_data)
            print('Node embeddings (GNN, GPU):', node_embeddings)

            # --- Link Prediction (previsão de novas conexões) ---
            node_map = {n: i for i, n in enumerate(G.nodes())}
            possible_pairs = [
                (node_map[u], node_map[v])
                for u in G.nodes() for v in G.nodes()
                if u != v and not G.has_edge(u, v)
            ]
            def link_prediction_scores(node_embeddings, pairs):
                scores = []
                for u, v in pairs:
                    score = (node_embeddings[u] * node_embeddings[v]).sum()
                    scores.append(score.item())
                return scores
            if possible_pairs:
                scores = link_prediction_scores(node_embeddings, possible_pairs)
                top_indices = torch.topk(torch.tensor(scores), min(5, len(scores))).indices
                for idx in top_indices:
                    u_idx, v_idx = possible_pairs[idx]
                    u = list(G.nodes())[u_idx]
                    v = list(G.nodes())[v_idx]
                    print(f"Sugerir nova conexão: {u} <-> {v} (score={scores[idx]:.4f})")

            # --- Classificação de arestas ---
            edge_classifier = EdgeClassifier(node_embeddings.shape[1], num_classes=3).to('cuda')
            # Se você já treinou, descomente:
            # edge_classifier.load_state_dict(torch.load('meu_edge_classifier.pth'))
            edge_classifier.eval()
            for u, v in G.edges():
                u_idx = node_map[u]
                v_idx = node_map[v]
                emb_u = node_embeddings[u_idx].unsqueeze(0)
                emb_v = node_embeddings[v_idx].unsqueeze(0)
                logits = edge_classifier(emb_u, emb_v)
                pred_class = logits.argmax(dim=1).item()
                print(f"Aresta {u} <-> {v}: classe prevista = {pred_class}")
    finally:
        monitor.clear_memory()

if __name__ == "__main__":
    asyncio.run(main())