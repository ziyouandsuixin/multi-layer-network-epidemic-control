import networkx as nx
import numpy as np
from collections import defaultdict
import community as community_louvain

class DynamicCommunityDetectionSimple:
    """简化版动态社区发现"""
    
    def __init__(self):
        self.community_evolution = {}
    
    def detect_cross_layer_communities(self, multilayer_network, method='louvain'):
        """检测跨层社区"""
        physical_communities = self._detect_layer_communities_simple(
            multilayer_network.physical_layer, method
        )
        information_communities = self._detect_layer_communities_simple(
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
    
    def _detect_layer_communities_simple(self, graph, method='louvain'):
        """简化版社区检测"""
        if method == 'louvain':
            partition = community_louvain.best_partition(graph)
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            communities_list = list(communities_dict.values())
            
        elif method == 'label_propagation':
            communities_generator = nx.algorithms.community.label_propagation_communities(graph)
            communities_list = [list(comm) for comm in communities_generator]
            
        elif method == 'greedy_modularity':
            communities_generator = nx.algorithms.community.greedy_modularity_communities(graph)
            communities_list = [list(comm) for comm in communities_generator]
            
        else:
            raise ValueError(f"不支持的算法: {method}")
        
        return SimpleCommunityResult(communities_list)
    
    def track_temporal_communities(self, temporal_networks, method='louvain'):
        """追踪时序社区演化"""
        temporal_communities = {}
        previous_communities = None
        
        for time_point, network in sorted(temporal_networks.items()):
            current_communities = self._detect_layer_communities_simple(network, method)
            
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

class SimpleCommunityResult:
    """简化的社区结果类"""
    def __init__(self, communities):
        self.communities = communities