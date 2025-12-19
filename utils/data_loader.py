import pandas as pd
import networkx as nx
import os
from tqdm import tqdm

class SimpleDataProcessor:
    """简化数据处理器"""
    
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)

class TemporalNetworkBuilder:
    """简化时序网络构建器"""
    
    def build_physical_network_from_mobility(self, mobility_data):
        """构建物理接触网络"""
        temporal_networks = {}
        
        for date in mobility_data['date'].unique():
            daily_data = mobility_data[mobility_data['date'] == date]
            G = nx.Graph()
            
            for _, row in daily_data.iterrows():
                G.add_node(row['individual_id'], **row.to_dict())
            
            location_groups = daily_data.groupby('exposure_location')
            for location, group in location_groups:
                individuals = group['individual_id'].tolist()
                
                for i in range(len(individuals)):
                    for j in range(i + 1, len(individuals)):
                        G.add_edge(individuals[i], individuals[j], 
                                 weight=1.0, location=location)
            
            temporal_networks[date] = G
        
        return temporal_networks
    
    def build_information_network(self, physical_networks, mobility_data):
        """构建信息传播网络"""
        information_networks = {}
        
        for date, phys_net in physical_networks.items():
            info_net = nx.Graph()
            
            for node, attr in phys_net.nodes(data=True):
                info_net.add_node(node, **attr)
            
            for node1 in phys_net.nodes():
                for node2 in phys_net.nodes():
                    if node1 != node2 and phys_net.has_edge(node1, node2):
                        info_net.add_edge(node1, node2, weight=0.5, type='information')
            
            information_networks[date] = info_net
        
        return information_networks