import networkx as nx
import numpy as np
from cdlib import algorithms
from collections import defaultdict

class DynamicCommunityDetection:
    """动态社区发现"""
    
    def __init__(self):
        self.community_evolution = {}
    
    def detect_cross_layer_communities(self, multilayer_network, method='leiden'):
        """检测跨层社区"""
        physical_communities = self._detect_layer_communities(
            multilayer_network.physical_layer, method
        )
        information_communities = self._detect_layer_communities(
            multilayer_network.information_layer, method
        )
        
        alignment_results = self._analyze_cross_layer_alignment(
            physical_communities, information_communities
        )
        
        multilayer_network.communities = {
            'physical': physical_communities,
            'information': information_communities,
            'alignment': alignment_results
        }
        
        return multilayer_network.communities
    
    def track_temporal_communities(self, temporal_networks, method='leiden'):
        """追踪时序社区演化"""
        temporal_communities = {}
        previous_communities = None
        
        for time_point, network in sorted(temporal_networks.items()):
            current_communities = self._detect_layer_communities(network, method)
            
            evolution_metrics = {}
            if previous_communities is not None:
                evolution_metrics = self._calculate_community_evolution(
                    previous_communities, current_communities
                )
            
            temporal_communities[time_point] = {
                'communities': current_communities,
                'evolution': evolution_metrics,
                'network': network
            }
            
            previous_communities = current_communities
        
        return temporal_communities
    
    def _detect_layer_communities(self, graph, method='leiden'):
        """检测单层网络社区"""
        if method == 'leiden':
            communities = algorithms.leiden(graph)
        elif method == 'louvain':
            communities = algorithms.louvain(graph)
        elif method == 'infomap':
            communities = algorithms.infomap(graph)
        else:
            raise ValueError(f"不支持的算法: {method}")
        
        return communities
    
    def _analyze_cross_layer_alignment(self, phys_comms, info_comms):
        """分析跨层社区对齐"""
        alignment = {
            'jaccard_similarity': 0.0,
            'high_risk_misalignment': [],
            'aligned_communities': []
        }
        
        phys_sets = [set(comm) for comm in phys_comms.communities]
        info_sets = [set(comm) for comm in info_comms.communities]
        
        total_similarity = 0.0
        pair_count = 0
        
        for i, phys_set in enumerate(phys_sets):
            for j, info_set in enumerate(info_sets):
                if phys_set and info_set:
                    jaccard = len(phys_set.intersection(info_set)) / len(phys_set.union(info_set))
                    total_similarity += jaccard
                    pair_count += 1
                    
                    if jaccard > 0.5:
                        alignment['aligned_communities'].append({
                            'physical_community': i,
                            'information_community': j,
                            'similarity': jaccard,
                            'size_physical': len(phys_set),
                            'size_information': len(info_set)
                        })
        
        if pair_count > 0:
            alignment['jaccard_similarity'] = total_similarity / pair_count
        
        return alignment
    
    def _calculate_community_evolution(self, prev_comms, current_comms):
        """计算社区演化指标"""
        evolution = {
            'community_similarity': 0.0,
            'survival_ratio': 0.0,
            'split_events': 0,
            'merge_events': 0
        }
        
        prev_nodes = set([node for comm in prev_comms.communities for node in comm])
        current_nodes = set([node for comm in current_comms.communities for node in comm])
        
        common_nodes = prev_nodes.intersection(current_nodes)
        if prev_nodes:
            evolution['survival_ratio'] = len(common_nodes) / len(prev_nodes)
        
        return evolution