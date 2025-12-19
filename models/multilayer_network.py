import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

class MultilayerEpidemicNetwork:
    """双层疫情传播网络 - 修复版"""
    
    def __init__(self):
        self.physical_network = None
        self.information_network = None
        self.node_mapping = {}
        self.communities = {}
    
    def build_physical_layer(self, mobility_data, exposure_data):
        """构建物理传播层 - 修复属性设置"""
        print("构建物理传播层...")
        G_physical = nx.Graph()
        
        for _, row in mobility_data.iterrows():
            node_id = row['individual_id']
            G_physical.add_node(node_id, **{
                'type': 'individual',
                'mobility': row.get('mobility', 'medium'),
                'location': row.get('location', 'unknown'),
                'layer': 'physical'
            })
        
        exposure_groups = exposure_data.groupby('location_id')
        for location, group in exposure_groups:
            individuals = group['case_id'].tolist()
            for i in range(len(individuals)):
                for j in range(i + 1, len(individuals)):
                    weight = self._calculate_physical_risk(group.iloc[i], group.iloc[j])
                    G_physical.add_edge(individuals[i], individuals[j], 
                                      weight=weight, 
                                      type='exposure_contact')
        
        self.physical_network = G_physical
        print(f"物理层: {G_physical.number_of_nodes()}节点, {G_physical.number_of_edges()}边")
        return G_physical
    
    def build_information_layer(self, physical_network, risk_perception_data=None):
        """构建信息感知层 - 修复属性设置"""
        print("构建信息感知层...")
        G_information = nx.Graph()
        
        for node, attr in physical_network.nodes(data=True):
            G_information.add_node(node, **attr)
            G_information.nodes[node]['layer'] = 'information'
            G_information.nodes[node]['risk_perception'] = np.random.uniform(0.1, 0.9)
            G_information.nodes[node]['information_activity'] = np.random.uniform(0.3, 1.0)
        
        nodes_list = list(physical_network.nodes())
        for i, node1 in enumerate(nodes_list):
            for j in range(i + 1, len(nodes_list)):
                node2 = nodes_list[j]
                info_prob = self._calculate_information_probability(
                    physical_network, node1, node2
                )
                if info_prob > 0.1:
                    G_information.add_edge(node1, node2, 
                                         weight=info_prob,
                                         type='information_flow')
        
        self.information_network = G_information
        print(f"信息层: {G_information.number_of_nodes()}节点, {G_information.number_of_edges()}边")
        return G_information
    
    def _calculate_physical_risk(self, exposure1, exposure2):
        """计算物理接触风险 - 增强实现"""
        risk = 0.0
        
        time_overlap = self._calculate_time_overlap(
            exposure1['exposure_start'], exposure1['exposure_end'],
            exposure2['exposure_start'], exposure2['exposure_end']
        )
        risk += time_overlap * 0.6
        
        risk_level_map = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
        risk1 = risk_level_map.get(exposure1.get('risk_level', 'low'), 0.1)
        risk2 = risk_level_map.get(exposure2.get('risk_level', 'low'), 0.1)
        risk += (risk1 + risk2) * 0.2
        
        duration1 = (exposure1['exposure_end'] - exposure1['exposure_start']).total_seconds()
        duration2 = (exposure2['exposure_end'] - exposure2['exposure_start']).total_seconds()
        avg_duration = (duration1 + duration2) / (2 * 3600)
        duration_factor = min(avg_duration / 24.0, 1.0)
        risk += duration_factor * 0.2
        
        return min(risk, 1.0)
    
    def _calculate_information_probability(self, physical_net, node1, node2):
        """计算信息传播概率 - 增强实现"""
        prob = 0.0
        
        if physical_net.has_edge(node1, node2):
            edge_weight = physical_net[node1][node2].get('weight', 0.5)
            prob += edge_weight * 0.5
        
        node1_loc = physical_net.nodes[node1].get('location', '')
        node2_loc = physical_net.nodes[node2].get('location', '')
        if node1_loc == node2_loc and node1_loc != 'unknown':
            prob += 0.3
        
        node1_mob = physical_net.nodes[node1].get('mobility', 'medium')
        node2_mob = physical_net.nodes[node2].get('mobility', 'medium')
        if node1_mob == node2_mob:
            prob += 0.15
        
        prob += np.random.uniform(0.0, 0.05)
        
        return min(prob, 1.0)
    
    def _calculate_time_overlap(self, start1, end1, start2, end2):
        """计算时间重叠 - 增强容错"""
        try:
            if isinstance(start1, str):
                start1 = datetime.fromisoformat(start1)
            if isinstance(end1, str):
                end1 = datetime.fromisoformat(end1)
            if isinstance(start2, str):
                start2 = datetime.fromisoformat(start2)
            if isinstance(end2, str):
                end2 = datetime.fromisoformat(end2)
            
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                total_duration = (max(end1, end2) - min(start1, start2)).total_seconds()
                return overlap_duration / total_duration if total_duration > 0 else 0
        except Exception as e:
            print(f"时间重叠计算错误: {e}")
        return 0.0
    
    def get_network_info(self):
        """获取网络信息 - 调试用"""
        info = {
            'physical_network': {
                'exists': self.physical_network is not None,
                'nodes': self.physical_network.number_of_nodes() if self.physical_network else 0,
                'edges': self.physical_network.number_of_edges() if self.physical_network else 0,
                'id': id(self.physical_network) if self.physical_network else None
            },
            'information_network': {
                'exists': self.information_network is not None,
                'nodes': self.information_network.number_of_nodes() if self.information_network else 0,
                'edges': self.information_network.number_of_edges() if self.information_network else 0,
                'id': id(self.information_network) if self.information_network else None
            }
        }
        return info
    
    def validate_networks(self):
        """验证网络完整性"""
        issues = []
        
        if self.physical_network is None:
            issues.append("物理层网络为 None")
        elif self.physical_network.number_of_nodes() == 0:
            issues.append("物理层网络节点数为 0")
        
        if self.information_network is None:
            issues.append("信息层网络为 None")
        elif self.information_network.number_of_nodes() == 0:
            issues.append("信息层网络节点数为 0")
        
        if (self.physical_network and self.information_network and
            self.physical_network.number_of_nodes() != self.information_network.number_of_nodes()):
            issues.append("两层网络节点数不一致")
        
        return issues