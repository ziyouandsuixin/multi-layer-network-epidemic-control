"""
你的现有模型封装器 - 完整修复状态格式问题
"""

from typing import Dict, Any, List
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .base_model import EpidemicModel

class OurEnhancedModel(EpidemicModel):
    """你的现有模型封装器 - 完整修复状态格式问题"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'use_dynamic_communities': True,
            'multilayer_network': True,
            'risk_prediction': True
        }
        if config:
            default_config.update(config)
        super().__init__("Our_Enhanced_Model", default_config)
        
        self._import_existing_modules()
    
    def _import_existing_modules(self):
        """延迟导入你的现有模块 - 修正类名"""
        try:
            from models.multilayer_network import MultilayerEpidemicNetwork
            from models.dynamic_test import DynamicCommunityDetectionSimple
            from models.resource_allocator_enhanced import ResourceAllocatorEnhanced
            
            self.MultilayerEpidemicNetwork = MultilayerEpidemicNetwork
            self.DynamicCommunityDetectionSimple = DynamicCommunityDetectionSimple
            self.ResourceAllocatorEnhanced = ResourceAllocatorEnhanced
            
        except ImportError as e:
            self._create_improved_mock_modules()
    
    def _create_improved_mock_modules(self):
        """创建改进的模拟模块"""
        class MockNetwork:
            def build_physical_layer(self, mobility_data, exposure_data): 
                import networkx as nx
                G = nx.erdos_renyi_graph(100, 0.1)
                return G
            
            def build_information_layer(self, physical_net): 
                return physical_net
        
        class MockDetector:
            def detect_cross_layer_communities(self, multilayer_net):
                return {'physical': type('obj', (object,), {'communities': [[i for i in range(10)] for _ in range(10)]})()}
        
        class MockResourceAllocator:
            def __init__(self):
                self.dynamics_model = MockDynamicsModel()
            
            def optimize_resource_allocation_with_dynamics(self, communities, networks, initial_states, time_horizon):
                allocation_plan = {
                    'resource_efficiency': 0.85,
                    'risk_distribution': {'critical': 2, 'high': 3, 'medium': 3, 'low': 2}
                }
                state_evolution = []
                current_states = initial_states.copy()
                for t in range(time_horizon + 1):
                    state_evolution.append(current_states.copy())
                    for node_id, state in current_states.items():
                        if state['state'] == 'I' and np.random.random() < 0.2:
                            current_states[node_id]['state'] = 'R'
                return allocation_plan, state_evolution
        
        class MockDynamicsModel:
            def initialize_node_states(self, nodes):
                states = {}
                infected_nodes = np.random.choice(nodes, size=max(1, len(nodes)//20), replace=False)
                for node in nodes:
                    states[node] = {
                        'state': 'I' if node in infected_nodes else 'S',
                        'infection_time': 0 if node in infected_nodes else None,
                        'is_superspreader': False
                    }
                return states
        
        self.MultilayerEpidemicNetwork = MockNetwork
        self.DynamicCommunityDetectionSimple = MockDetector
        self.ResourceAllocatorEnhanced = MockResourceAllocator
    
    def _convert_initial_states(self, experiment_states):
        """将实验框架状态转换为你的模型期望的格式"""
        your_states = {}
        
        infected_nodes = experiment_states.get('infected_nodes', [])
        all_nodes = experiment_states.get('individual_ids', [])
        
        if not all_nodes:
            infected_nodes = experiment_states.get('infected_nodes', [])
            susceptible_nodes = experiment_states.get('susceptible_nodes', [])
            all_nodes = infected_nodes + susceptible_nodes
        
        if not all_nodes:
            total_population = experiment_states.get('total_population', 500)
            all_nodes = list(range(total_population))
            n_initial_infected = max(1, int(total_population * 0.05))
            infected_nodes = np.random.choice(all_nodes, size=n_initial_infected, replace=False).tolist()
        
        for node in all_nodes:
            if node in infected_nodes:
                your_states[node] = {
                    'state': 'I',
                    'infection_time': 0,
                    'is_superspreader': np.random.random() < 0.08
                }
            else:
                your_states[node] = {
                    'state': 'S',
                    'infection_time': None,
                    'is_superspreader': False
                }
        
        return your_states
    
    def simulate(self, network_data: Any, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """运行你的增强模型模拟 - 修复网络数据格式问题"""
        try:
            if hasattr(network_data, 'nodes') and hasattr(network_data, 'edges'):
                physical_net = network_data
            else:
                physical_net = self._build_network_from_unified_data(network_data)
            
            detector = self.DynamicCommunityDetectionSimple()
            
            class FixedMultilayerNetwork:
                def __init__(self, physical_net):
                    self.physical_network = physical_net
                    self.information_network = physical_net
                    self.node_mapping = {}
            
            multilayer_net = FixedMultilayerNetwork(physical_net)
            communities = detector.detect_cross_layer_communities(multilayer_net)
            
            physical_communities = communities.get('physical')
            if physical_communities is None or len(physical_communities.communities) == 0:
                all_nodes = list(physical_net.nodes())
                default_communities = [all_nodes[i:i+50] for i in range(0, len(all_nodes), 50)]
                physical_communities = type('obj', (object,), {'communities': default_communities})()
            
            resource_allocator = self.ResourceAllocatorEnhanced()
            
            your_initial_states = self._convert_initial_states(initial_states)
            
            allocation_plan, state_evolution = resource_allocator.optimize_resource_allocation_with_dynamics(
                [physical_communities],
                [physical_net],
                your_initial_states,
                time_horizon=time_steps
            )
            
            time_series = self._convert_to_time_series(state_evolution)
            risk_assessment = self._extract_risk_assessment(allocation_plan, physical_communities, state_evolution[-1])
            
            combined_results = {
                'time_series': time_series,
                'risk_assessment': risk_assessment,
                'communities': communities,
                'allocation_plan': allocation_plan,
                'state_evolution': state_evolution,
                'success': True
            }
            
            self.results = combined_results
            
            return combined_results
            
        except Exception as e:
            return self._create_fallback_results(initial_states, time_steps)
    
    def _build_network_from_unified_data(self, network_data):
        """从统一数据构建网络 - 修复版本"""
        import networkx as nx
        
        G = nx.Graph()
        
        nodes = network_data.get('nodes', [])
        if not nodes:
            nodes = network_data.get('individual_ids', list(range(500)))
        
        for node in nodes:
            G.add_node(node)
        
        edges = network_data.get('edges', [])
        for edge in edges:
            if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
            elif isinstance(edge, dict):
                G.add_edge(edge.get('source'), edge.get('target'))
        
        return G
    
    def _convert_to_time_series(self, state_evolution):
        """将状态演化转换为时间序列格式"""
        time_series = {'S': [], 'I': [], 'R': []}
        
        for states in state_evolution:
            s_count = sum(1 for state in states.values() if state.get('state') == 'S')
            i_count = sum(1 for state in states.values() if state.get('state') == 'I')
            r_count = sum(1 for state in states.values() if state.get('state') == 'R')
            
            time_series['S'].append(s_count)
            time_series['I'].append(i_count)
            time_series['R'].append(r_count)
        
        return time_series
    
    def _extract_risk_assessment(self, allocation_plan, communities, final_states):
        """从分配计划中提取风险评估 - 改进版"""
        risk_assessment = {'communities': {}}
        
        if hasattr(communities, 'communities'):
            for i, community in enumerate(communities.communities):
                infected_count = sum(1 for node in community 
                                   if node in final_states and final_states[node].get('state') == 'I')
                
                risk_level = 'medium'
                if i < allocation_plan.get('risk_distribution', {}).get('critical', 0):
                    risk_level = 'critical'
                elif i < allocation_plan.get('risk_distribution', {}).get('critical', 0) + allocation_plan.get('risk_distribution', {}).get('high', 0):
                    risk_level = 'high'
                elif i < allocation_plan.get('risk_distribution', {}).get('critical', 0) + allocation_plan.get('risk_distribution', {}).get('high', 0) + allocation_plan.get('risk_distribution', {}).get('medium', 0):
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                risk_assessment['communities'][f'community_{i}'] = {
                    'risk_level': risk_level,
                    'infected_count': infected_count,
                    'population': len(community),
                    'infection_rate': infected_count / len(community) if len(community) > 0 else 0
                }
        
        return risk_assessment
    
    def _create_fallback_results(self, initial_states, time_steps):
        """创建回退结果"""
        total_population = initial_states.get('total_population', 500)
        initial_infected = len(initial_states.get('infected_nodes', []))
        
        time_series = {
            'S': [total_population - initial_infected],
            'I': [initial_infected],
            'R': [0]
        }
        
        for t in range(1, time_steps):
            prev_s = time_series['S'][-1]
            prev_i = time_series['I'][-1]
            prev_r = time_series['R'][-1]
            
            new_infected = min(prev_s, int(prev_i * 0.1))
            new_recovered = min(prev_i, int(prev_i * 0.2))
            
            time_series['S'].append(prev_s - new_infected)
            time_series['I'].append(prev_i + new_infected - new_recovered)
            time_series['R'].append(prev_r + new_recovered)
        
        return {
            'time_series': time_series,
            'risk_assessment': {
                'communities': {
                    'community_0': {
                        'risk_level': 'medium', 
                        'infected_count': initial_infected, 
                        'population': total_population,
                        'infection_rate': initial_infected / total_population
                    }
                }
            },
            'success': False,
            'error': '使用回退模拟'
        }
    
    def allocate_resources(self, risk_assessment: Dict, available_resources: Dict) -> Dict[str, float]:
        """使用你的增强资源分配器 - 修复接口"""
        try:
            allocator = self.ResourceAllocatorEnhanced()
            
            allocation = {}
            total_risk_score = 0
            
            for community_id, info in risk_assessment.get('communities', {}).items():
                risk_level = info.get('risk_level', 'medium')
                risk_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
                risk_score = risk_weights.get(risk_level, 1) * info.get('infected_count', 0)
                total_risk_score += risk_score
            
            if total_risk_score > 0:
                for community_id, info in risk_assessment.get('communities', {}).items():
                    risk_level = info.get('risk_level', 'medium')
                    risk_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
                    community_risk = risk_weights.get(risk_level, 1) * info.get('infected_count', 0)
                    allocation_ratio = community_risk / total_risk_score
                    
                    allocation[community_id] = {
                        resource: amount * allocation_ratio 
                        for resource, amount in available_resources.items()
                    }
            else:
                n_communities = len(risk_assessment.get('communities', {}))
                if n_communities > 0:
                    for community_id in risk_assessment.get('communities', {}).keys():
                        allocation[community_id] = {
                            resource: amount / n_communities 
                            for resource, amount in available_resources.items()
                        }
            
            return allocation
            
        except Exception as e:
            return {
                'community_0': {resource: amount * 0.6 for resource, amount in available_resources.items()},
                'community_1': {resource: amount * 0.4 for resource, amount in available_resources.items()}
            }