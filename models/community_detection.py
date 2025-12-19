import networkx as nx
import numpy as np
from collections import defaultdict
import community as community_louvain

class StaticCommunityDetection:
    """静态社区发现算法"""
    
    def __init__(self):
        self.methods = {
            'louvain': self._louvain_communities,
            'label_propagation': self._label_propagation,
            'greedy_modularity': self._greedy_modularity
        }
    
    def detect_communities(self, graph, method='louvain', **kwargs):
        """社区发现主函数"""
        if method not in self.methods:
            raise ValueError(f"不支持的方法: {method}，可用方法: {list(self.methods.keys())}")
        
        return self.methods[method](graph, **kwargs)
    
    def _louvain_communities(self, graph, **kwargs):
        """Louvain算法"""
        partition = community_louvain.best_partition(graph)
        
        communities_dict = defaultdict(list)
        for node, comm_id in partition.items():
            communities_dict[comm_id].append(node)
        
        communities_list = list(communities_dict.values())
        return SimpleCommunityResult(communities_list)
    
    def _label_propagation(self, graph, **kwargs):
        """标签传播算法"""
        communities_generator = nx.algorithms.community.label_propagation_communities(graph)
        communities_list = [list(comm) for comm in communities_generator]
        return SimpleCommunityResult(communities_list)
    
    def _greedy_modularity(self, graph, **kwargs):
        """贪心模块度算法"""
        communities_generator = nx.algorithms.community.greedy_modularity_communities(graph)
        communities_list = [list(comm) for comm in communities_generator]
        return SimpleCommunityResult(communities_list)
    
    def evaluate_communities(self, graph, communities):
        """评估社区质量"""
        if hasattr(communities, 'communities'):
            comm_list = communities.communities
        else:
            comm_list = communities
        
        if graph.number_of_edges() == 0:
            metrics = {
                'modularity': 0.0,
                'n_communities': len(comm_list),
                'avg_community_size': np.mean([len(comm) for comm in comm_list]) if comm_list else 0,
                'coverage': 1.0 if comm_list else 0.0,
                'network_edges': 0,
                'network_nodes': graph.number_of_nodes()
            }
            return metrics
        
        try:
            modularity = nx.algorithms.community.quality.modularity(graph, comm_list)
        except ZeroDivisionError:
            modularity = 0.0
        
        metrics = {
            'modularity': modularity,
            'n_communities': len(comm_list),
            'avg_community_size': np.mean([len(comm) for comm in comm_list]) if comm_list else 0,
            'coverage': self._calculate_coverage(graph, comm_list),
            'network_edges': graph.number_of_edges(),
            'network_nodes': graph.number_of_nodes()
        }
        return metrics
    
    def _calculate_coverage(self, graph, communities):
        """计算社区覆盖度"""
        nodes_in_communities = set()
        for comm in communities:
            nodes_in_communities.update(comm)
        
        return len(nodes_in_communities) / len(graph.nodes) if graph.nodes else 0

class SimpleCommunityResult:
    """简化的社区结果类"""
    def __init__(self, communities):
        self.communities = communities

class DynamicCommunityDetectionSimple:
    """简化版动态社区发现"""
    
    def __init__(self):
        self.static_detector = StaticCommunityDetection()
    
    def detect_temporal_communities(self, temporal_networks, method='louvain'):
        """时序社区发现"""
        temporal_communities = {}
        
        for time_point, graph in temporal_networks.items():
            communities = self.static_detector.detect_communities(graph, method)
            metrics = self.static_detector.evaluate_communities(graph, communities)
            
            temporal_communities[time_point] = {
                'communities': communities,
                'metrics': metrics,
                'graph': graph
            }
        
        return temporal_communities
    
    def detect_high_risk_communities(self, temporal_communities, risk_factors):
        """识别高风险社区"""
        high_risk_communities = {}
        
        for time_point, data in temporal_communities.items():
            communities = data['communities']
            graph = data['graph']
            
            community_risks = []
            
            for i, community in enumerate(communities.communities):
                risk_score = self._calculate_community_risk(community, graph, risk_factors)
                community_risks.append({
                    'community_id': i,
                    'nodes': list(community),
                    'risk_score': risk_score,
                    'size': len(community)
                })
            
            community_risks.sort(key=lambda x: x['risk_score'], reverse=True)
            high_risk_communities[time_point] = community_risks
        
        return high_risk_communities
    
    def track_community_evolution(self, temporal_communities):
        """追踪社区演化（简化版）"""
        evolution_data = {}
        
        time_points = sorted(temporal_communities.keys())
        
        for i in range(1, len(time_points)):
            current_time = time_points[i]
            prev_time = time_points[i-1]
            
            current_comms = temporal_communities[current_time]['communities']
            prev_comms = temporal_communities[prev_time]['communities']
            
            similarity = self._calculate_community_similarity(prev_comms, current_comms)
            evolution_data[current_time] = {
                'similarity_score': similarity,
                'prev_communities': len(prev_comms.communities),
                'current_communities': len(current_comms.communities)
            }
        
        return evolution_data
    
    def _calculate_community_risk(self, community, graph, risk_factors):
        """计算社区风险分数"""
        risk_score = 0.0
        
        for node in community:
            node_data = graph.nodes[node]
            
            if node_data.get('data_type') == 'exposure':
                risk_score += 2.0
            if node_data.get('risk_factor', 0) > 0.5:
                risk_score += 1.5
            if node_data.get('location') and 'transport' in str(node_data.get('location')):
                risk_score += 1.0
        
        risk_score = risk_score / len(community) if community else 0
        
        return risk_score
    
    def _calculate_community_similarity(self, comms1, comms2):
        """计算社区相似性（简化版）"""
        nodes1 = set([node for comm in comms1.communities for node in comm])
        nodes2 = set([node for comm in comms2.communities for node in comm])
        
        if not nodes1 or not nodes2:
            return 0.0
        
        intersection = nodes1.intersection(nodes2)
        union = nodes1.union(nodes2)
        
        return len(intersection) / len(union) if union else 0.0