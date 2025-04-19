# graph_renderer.py - Utility functions for generating visualizations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import json
from typing import Dict, List, Any, Tuple, Optional

def create_process_flow_graph(events_data: List[Dict[str, Any]]) -> str:
    """
    Creates a process flow graph visualization from event data.
    
    Args:
        events_data: List of event dictionaries containing 'case_id', 'activity', and 'timestamp'
        
    Returns:
        Base64 encoded PNG image string
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(events_data)
    
    # Ensure required columns exist
    required_cols = ['case_id', 'activity', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Event data must contain columns: {required_cols}")
    
    # Ensure timestamp is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by case_id and extract transitions
    edges_count = {}
    
    for case_id, case_group in df.groupby('case_id'):
        # Sort by timestamp
        case_group = case_group.sort_values('timestamp')
        activities = case_group['activity'].tolist()
        
        # Add edges between consecutive activities
        for i in range(len(activities) - 1):
            source = activities[i]
            target = activities[i + 1]
            edge = (source, target)
            
            if edge in edges_count:
                edges_count[edge] += 1
            else:
                edges_count[edge] = 1
    
    # Add edges to graph with weights
    for (source, target), weight in edges_count.items():
        G.add_edge(source, target, weight=weight)
    
    # Normalize edge weights for visualization
    max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
    for u, v in G.edges():
        G[u][v]['norm_weight'] = G[u][v]['weight'] / max_weight
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Consistent layout for same data
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='skyblue', 
                          node_size=800,
                          alpha=0.8,
                          edgecolors='black')
    
    # Draw edges with varying widths based on weight
    edge_widths = [G[u][v]['norm_weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=edge_widths,
                          edge_color='gray',
                          alpha=0.7,
                          arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels (frequencies)
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add title and adjust layout
    plt.title("Process Flow Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Convert to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode()

def create_histogram(data: List[float], title: str, xlabel: str, ylabel: str) -> str:
    """
    Creates a histogram visualization.
    
    Args:
        data: List of numerical values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Base64 encoded PNG image string
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    
    # Convert to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode()

def create_bar_chart(categories: List[str], values: List[float], title: str, xlabel: str, ylabel: str) -> str:
    """
    Creates a bar chart visualization.
    
    Args:
        categories: List of category labels
        values: List of values for each category
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Base64 encoded PNG image string
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by values if there are more than a few categories
    if len(categories) > 5:
        sorted_indices = np.argsort(values)[::-1]  # Descending order
        categories = [categories[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
    
    # Create horizontal bar chart for better readability with many categories
    if len(categories) > 10:
        plt.barh(categories, values, color='steelblue', alpha=0.8)
        plt.xlabel(ylabel)  # Swap x and y labels for horizontal
        plt.ylabel(xlabel)
    else:
        plt.bar(categories, values, color='steelblue', alpha=0.8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45 if len(categories) > 5 else 0)
    
    plt.title(title, fontsize=14)
    plt.grid(axis='x' if len(categories) > 10 else 'y', alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode()

def create_knowledge_graph(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
    """
    Creates a knowledge graph visualization.
    
    Args:
        entities: List of entity dictionaries with 'id' and 'type'
        relationships: List of relationship dictionaries with 'source', 'target', and 'type'
        
    Returns:
        Base64 encoded PNG image string
    """
    G = nx.Graph()
    
    # Add nodes with types as attributes
    entity_types = {}
    for entity in entities:
        if 'id' not in entity:
            # Use first key-value pair as id if not specified
            entity_id = f"{list(entity.keys())[0]}_{entity[list(entity.keys())[0]]}"
        else:
            entity_id = entity['id']
            
        entity_type = entity.get('type', 'unknown')
        G.add_node(entity_id, type=entity_type)
        entity_types[entity_id] = entity_type
    
    # Add edges with types as attributes
    for rel in relationships:
        G.add_edge(rel['source'], rel['target'], type=rel.get('type', ''))
    
    # Create the visualization
    plt.figure(figsize=(15, 10))
    
    # Use different colors for different entity types
    unique_types = set(entity_types.values())
    color_map = {t: plt.cm.tab10(i) for i, t in enumerate(unique_types)}
    
    # Node colors based on type
    node_colors = [color_map[entity_types[node]] for node in G.nodes()]
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8, edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9)
    
    # Create legend for node types
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[t], markersize=10, 
                                label=t) for t in unique_types]
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.title("Knowledge Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Convert to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode()