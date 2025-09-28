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

def setup_logging():
    """Configure optimized logging with file and console output."""
    log_filename = f"gnn_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    handlers = [
        logging.FileHandler(log_filename, buffering=1024),
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
        self.batch_size = 5000  # Increased batch size for better performance
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
        USING PERIODIC COMMIT 1000
        UNWIND $batch AS node
        CREATE (n:Node)
        SET n = node
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
        USING PERIODIC COMMIT 1000
        UNWIND $batch AS edge
        MATCH (source:Node {id: edge.source_id})
        USING INDEX source:Node(id)
        MATCH (target:Node {id: edge.target_id})
        USING INDEX target:Node(id)
        CREATE (source)-[r:CONNECTS_TO]->(target)
        SET r = edge.properties
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
            uri=neo4j_config.get("uri", "neo4j://localhost:7687"),
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", "your_password")
        )
    except Exception as e:
        logger.error(f"Failed to export to Neo4j: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process network topology file and export to Neo4j')
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (KML or KMZ)',
        required=True
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='neo4j://localhost:7687',
        help='Neo4j database URI (default: neo4j://localhost:7687)'
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
        required=True,
        help='Neo4j password'
    )
    
    return parser.parse_args()

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
        
        # Log initial resource usage
        monitor.log_usage()
        
        try:
            # Create network builder
            builder = NetworkBuilder()
            logger.info("Created NetworkBuilder instance")
            
            # Load and process file in chunks
            xml_data = load_kml_content(args.input)
            logger.info("Successfully loaded KML content")
            monitor.log_usage()
            
            # Process XML in parallel chunks
            chunks = split_xml_content(xml_data)
            logger.info(f"Split content into {len(chunks)} chunks for processing")
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Process chunks in parallel
                futures = []
                for i, chunk in enumerate(chunks):
                    future = executor.submit(process_kml_chunk, chunk)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())
                    monitor.log_usage()
            
            # Create base OLT
            builder.create_olt()
            
            # Build graph from processed results
            for node_data in results:
                if node_data['type'] == 'ONU':
                    builder.create_onu(
                        node_data['id'],
                        node_data['coords'],
                        node_data.get('description', '')
                    )
            
            # Update status with optimized batch processing
            builder.bulk_update_status()
            logger.info("Updated node statuses")
            monitor.log_usage()
            
            # Get the graph
            G = builder.get_graph()
            logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Export to Neo4j with optimized settings
            neo4j_config = {
                "uri": args.neo4j_uri,
                "username": args.neo4j_user,
                "password": args.neo4j_password
            }
            
            # Run the Neo4j export
            await export_graph_to_neo4j(G, neo4j_config)
            logger.info("Successfully exported graph to Neo4j")
            
            # Final resource usage and timing
            monitor.log_usage()
            duration = datetime.now() - start_time
            logger.info(f"Total processing time: {duration}")
            
        finally:
            # Clean up resources
            monitor.clear_memory()
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())