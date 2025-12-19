#!/usr/bin/env python3
"""
增强版资源分配器 - 包含动力学功能（完整修复版）
修复网络对象处理和社区数据结构问题
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx

class CommunityStructure:
    """社区数据结构封装"""
    def __init__(self, communities):
        self.communities = communities
        self.community_count = len(communities)
    
    def get_all_nodes(self):
        """获取所有节点"""
        all_nodes = []
        for community in self.communities:
            all_nodes.extend(community)
        return list(set(all_nodes))

class CommunityEpidemicDynamics:
    """社区内部流行病动力学 - 修复版"""
    
    def __init__(self, beta=0.03, gamma=0.5, incubation_period=5.2):
        self.beta = beta
        self.gamma = gamma  
        self.incubation_period = incubation_period
        self.superspreader_prob = 0.08
        self.superspreader_factor = 3.0
        
    def initialize_node_states(self, nodes, initial_infected_ratio=0.05):
        """初始化节点状态"""
        states = {}
        n_infected = max(1, int(len(nodes) * initial_infected_ratio))
        
        for i, node in enumerate(nodes):
            if i < n_infected:
                states[node] = {
                    'state': 'I',
                    'infection_time': 0,
                    'is_superspreader': np.random.random() < self.superspreader_prob
                }
            else:
                states[node] = {
                    'state': 'S', 
                    'infection_time': None,
                    'is_superspreader': False
                }
        
        print(f"初始化状态: {len(states)}个节点, {n_infected}个感染者")
        return states
    
    def update_community_states(self, community_nodes, current_states, contact_network, time_step):
        """更新社区内部状态 - 修复深拷贝问题"""
        new_states = {}
        for node, state_data in current_states.items():
            new_states[node] = {
                'state': state_data['state'],
                'infection_time': state_data['infection_time'],
                'is_superspreader': state_data['is_superspreader']
            }
        
        initial_infected = sum(1 for node in community_nodes 
                            if node in new_states and new_states[node]['state'] == 'I')
        
        print(f"时间步 {time_step} 状态更新:")
        print(f"   初始感染数: {initial_infected}/{len(community_nodes)}")
        
        for node in community_nodes:
            if node not in new_states:
                new_states[node] = {
                    'state': 'S',
                    'infection_time': None,
                    'is_superspreader': False
                }
        
        infection_count = 0
        recovery_count = 0
        
        for node in community_nodes:
            node_data = new_states[node]
            old_state = node_data['state']
            
            if node_data['state'] == 'S':
                self._susceptible_dynamics(node, node_data, community_nodes, new_states, contact_network, time_step)
                if node_data['state'] == 'I':
                    infection_count += 1
                    
            elif node_data['state'] == 'I':
                old_infection_time = node_data['infection_time']
                self._infected_dynamics(node, node_data, time_step)
                if node_data['state'] == 'S':
                    recovery_count += 1
        
        final_infected = sum(1 for node in community_nodes 
                            if node in new_states and new_states[node]['state'] == 'I')
        
        print(f"   新感染: {infection_count}, 恢复: {recovery_count}")
        print(f"   最终感染数: {final_infected}/{len(community_nodes)}")
        print(f"   感染变化: {final_infected - initial_infected:+d}")
        
        return new_states
    
    def _susceptible_dynamics(self, node, node_data, community_nodes, states, contact_network, time_step):
        if contact_network is None:
            return
            
        try:
            infection_pressure = self._calculate_infection_pressure(node, states, contact_network)
            
            if infection_pressure > 0:
                base_prob = 1 - (1 - self.beta) ** infection_pressure
                
                superspreader_boost = 1.0
                superspreader_count = 0
                
                for neighbor in contact_network.neighbors(node):
                    if (neighbor in states and states[neighbor]['state'] == 'I' 
                        and states[neighbor]['is_superspreader']):
                        superspreader_count += 1
                
                if superspreader_count > 0:
                    superspreader_boost = min(1.0 + (superspreader_count * 0.2), 2.0)
                
                infection_prob = min(base_prob * superspreader_boost, 0.95)
                
                if np.random.random() < infection_prob:
                    print(f"     节点 {node} 被感染 (压力: {infection_pressure}, 概率: {infection_prob:.3f})")
                    node_data['state'] = 'I'
                    node_data['infection_time'] = time_step
                    node_data['is_superspreader'] = np.random.random() < self.superspreader_prob
        except Exception as e:
            print(f"易感者动力学计算失败: {e}")

    def _infected_dynamics(self, node, node_data, time_step):
        """感染者动力学 - 添加恢复日志"""
        if node_data['infection_time'] is None:
            return
            
        infection_duration = time_step - node_data['infection_time']
        recovery_prob = min(self.gamma * infection_duration / 7.0, 0.8)
        
        if np.random.random() < recovery_prob:
            print(f"     节点 {node} 恢复 (感染时长: {infection_duration}, 概率: {recovery_prob:.3f})")
            node_data['state'] = 'S'
            node_data['infection_time'] = None
            node_data['is_superspreader'] = False
    
    def _calculate_infection_pressure(self, node, states, contact_network):
        """计算感染压力 - 修复网络访问"""
        if contact_network is None:
            return 0
            
        pressure = 0
        try:
            for neighbor in contact_network.neighbors(node):
                if neighbor in states and states[neighbor]['state'] == 'I':
                    pressure += 1.0
                    if states[neighbor]['is_superspreader']:
                        pressure += self.superspreader_factor - 1
        except Exception as e:
            print(f"计算感染压力失败: {e}")
            
        return pressure
    
    def calculate_community_risk(self, community_nodes, states, contact_network, community_id="unknown"):
        """计算社区风险 - 优化版，包含传播潜力和网络特征"""
        if not community_nodes:
            return 0.0
        
        try:
            infected_count = sum(1 for node in community_nodes 
                            if node in states and states[node]['state'] == 'I')
            infection_ratio = infected_count / len(community_nodes)
            
            total_pressure = 0
            for node in community_nodes:
                if node in states:
                    total_pressure += self._calculate_infection_pressure(node, states, contact_network)
            
            pressure_score = min(total_pressure / (len(community_nodes) * 10), 1.0)
            
            superspreader_nodes = []
            for node in community_nodes:
                if (node in states and states[node]['state'] == 'I' 
                    and states[node]['is_superspreader']):
                    superspreader_nodes.append(node)
            
            has_superspreader = len(superspreader_nodes) > 0
            
            transmission_potential = self._calculate_transmission_potential(community_nodes, contact_network)
            
            connectivity = self._calculate_community_connectivity(community_nodes, contact_network)
            
            superspreader_impact = self._calculate_superspreader_impact(community_nodes, states)
            
            risk_components = [
                infection_ratio * 0.2,
                transmission_potential * 0.4,
                connectivity * 0.2,
                superspreader_impact * 0.2
            ]
            
            final_risk = min(sum(risk_components), 1.0)
            
            final_risk = self._apply_size_normalization(final_risk, len(community_nodes))
            
            print(f"社区风险分析 [ID: {community_id}]:")
            print(f"   社区大小: {len(community_nodes)} 节点")
            print(f"   感染情况: {infected_count}/{len(community_nodes)} = {infection_ratio:.3f}")
            print(f"   感染压力: {total_pressure} → 标准化: {pressure_score:.3f}")
            print(f"   超级传播者: {has_superspreader} ({len(superspreader_nodes)}个)")
            print(f"   传播潜力: {transmission_potential:.3f}")
            print(f"   连通性: {connectivity:.3f}")
            print(f"   超级传播影响: {superspreader_impact:.3f}")
            print(f"   风险分量: 感染{risk_components[0]:.3f} + 传播潜力{risk_components[1]:.3f} + 连通性{risk_components[2]:.3f} + 超级传播{risk_components[3]:.3f}")
            print(f"   总风险分数: {final_risk:.3f}")
            
            return final_risk
        except Exception as e:
            print(f"计算社区风险失败: {e}")
            return 0.5

    def _calculate_transmission_potential(self, community_nodes, contact_network):
        """计算传播潜力 - 考虑网络中心性"""
        if not contact_network or len(community_nodes) == 0:
            return 0.5
        
        try:
            degrees = []
            for node in community_nodes:
                if node in contact_network:
                    degrees.append(contact_network.degree(node))
            
            if not degrees:
                return 0.5
                
            avg_degree = np.mean(degrees)
            max_degree = max(degrees)
            
            transmission_potential = avg_degree / max(1, max_degree)
            return min(transmission_potential, 1.0)
            
        except Exception as e:
            print(f"计算传播潜力失败: {e}")
            return 0.5

    def _calculate_community_connectivity(self, community_nodes, contact_network):
        """计算社区内部连通性"""
        if not contact_network or len(community_nodes) < 2:
            return 0.5
        
        try:
            subgraph_nodes = [node for node in community_nodes if node in contact_network]
            if len(subgraph_nodes) < 2:
                return 0.5
                
            subgraph = contact_network.subgraph(subgraph_nodes)
            
            clustering_coeffs = nx.clustering(subgraph)
            avg_clustering = np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0
            
            return min(avg_clustering, 1.0)
            
        except Exception as e:
            print(f"计算社区连通性失败: {e}")
            return 0.5

    def _calculate_superspreader_impact(self, community_nodes, states):
        """计算超级传播者影响 - 考虑比例而非存在性"""
        try:
            superspreader_count = sum(1 for node in community_nodes 
                                    if node in states and states[node]['is_superspreader'])
            
            if len(community_nodes) == 0:
                return 0.1
                
            superspreader_ratio = superspreader_count / len(community_nodes)
            
            impact = min(superspreader_ratio * 5, 1.0)
            
            return max(impact, 0.1)
            
        except Exception as e:
            print(f"计算超级传播者影响失败: {e}")
            return 0.3

    def _apply_size_normalization(self, risk_score, community_size):
        """对小社区给予适当权重提升，避免规模偏差"""
        if community_size < 30:
            normalized_risk = min(risk_score * 1.4, 1.0)
        elif community_size < 100:
            normalized_risk = min(risk_score * 1.2, 1.0)
        elif community_size > 500:
            normalized_risk = risk_score * 0.9
        else:
            normalized_risk = risk_score
            
        return normalized_risk

class ResourceAllocatorEnhanced:
    """增强版资源分配器 - 完整实现"""
    
    def __init__(self):
        self.dynamics_model = CommunityEpidemicDynamics()
        self.resource_types = {
            'medical': ['vaccines', 'test_kits', 'medical_staff'],
            'non_medical': ['masks', 'sanitizers', 'information_materials']
        }
    
    def optimize_resource_allocation_with_dynamics(self, communities, contact_networks, 
                                                 initial_states=None, time_horizon=5):
        """增强版资源分配：结合动力学模拟"""
        print("执行动力学增强的资源分配...")
        
        state_evolution = self._simulate_epidemic_evolution(
            communities, contact_networks, initial_states, time_horizon
        )
        
        risk_predictions = self._predict_future_risks(communities, state_evolution, contact_networks)
        
        allocation_plan = self._optimize_dynamic_allocation(communities, risk_predictions)
        
        return allocation_plan, state_evolution
    
    def _simulate_epidemic_evolution(self, communities, contact_networks, initial_states, time_horizon):
        """模拟疫情演化 - 修复状态传递问题"""
        state_evolution = []
        
        current_states = {}
        for node, state_data in initial_states.items():
            current_states[node] = {
                'state': state_data['state'],
                'infection_time': state_data['infection_time'],
                'is_superspreader': state_data['is_superspreader']
            }
        
        state_evolution.append(current_states)
        
        print(f"开始模拟演化，时间范围: {time_horizon}步，初始节点数: {len(current_states)}")
        initial_infected = sum(1 for state in current_states.values() if state['state'] == 'I')
        print(f"初始感染数: {initial_infected}")
        
        for t in range(1, time_horizon + 1):
            print(f"时间步 {t} 开始 ====================")
            
            current_states_for_step = {}
            for node, state_data in current_states.items():
                current_states_for_step[node] = {
                    'state': state_data['state'],
                    'infection_time': state_data['infection_time'],
                    'is_superspreader': state_data['is_superspreader']
                }
            
            for i, community in enumerate(communities):
                community_nodes = self._extract_community_nodes(community)
                contact_net = contact_networks[i] if i < len(contact_networks) else None
                
                print(f"社区 {i} 更新:")
                for nodes in community_nodes:
                    current_states_for_step = self.dynamics_model.update_community_states(
                        nodes, current_states_for_step, contact_net, t
                    )
            
            state_evolution.append(current_states_for_step)
            current_states = current_states_for_step
            
            infected_count = sum(1 for state in current_states.values() if state['state'] == 'I')
            print(f"时间步 {t} 完成 - 总感染节点数: {infected_count}")
        
        return state_evolution
    
    def _extract_all_nodes(self, communities):
        """提取所有节点"""
        all_nodes = []
        for community in communities:
            community_nodes = self._extract_community_nodes(community)
            for nodes in community_nodes:
                all_nodes.extend(nodes)
        return list(set(all_nodes))
    
    def _extract_community_nodes(self, community):
        """统一提取社区节点"""
        if hasattr(community, 'communities'):
            return community.communities
        elif isinstance(community, dict) and 'nodes' in community:
            return [community['nodes']]
        elif isinstance(community, list) and all(isinstance(node, (str, int)) for node in community):
            return [community]
        else:
            print(f"未知的社区格式: {type(community)}")
            return [[]]
    
    def _predict_future_risks(self, communities, state_evolution, contact_networks):
        risk_predictions = {}
        
        print(f"验证状态演化数据:")
        for t, states in enumerate(state_evolution):
            infected_count = sum(1 for state in states.values() if state['state'] == 'I')
            print(f"  时间步 {t}: 感染数 = {infected_count}")
        
        for i, community in enumerate(communities):
            community_nodes_list = self._extract_community_nodes(community)
            contact_net = contact_networks[i] if i < len(contact_networks) else None
            
            for j, community_nodes in enumerate(community_nodes_list):
                comm_id = f"{i}_{j}"
                
                time_risks = []
                for t, states in enumerate(state_evolution):
                    comm_infected = sum(1 for node in community_nodes 
                                    if node in states and states[node]['state'] == 'I')
                    print(f"时间步 {t} - 社区 {comm_id}: {comm_infected}/{len(community_nodes)} 感染")
                    
                    risk = self.dynamics_model.calculate_community_risk(
                        community_nodes, states, contact_net, comm_id
                    )
                    time_risks.append(risk)
                
                future_risk = self._extrapolate_risk_trend(time_risks)
                risk_predictions[comm_id] = {
                    'current_risk': time_risks[-1],
                    'predicted_risk': future_risk,
                    'risk_trend': 'increasing' if future_risk > time_risks[-1] else 'decreasing',
                    'risk_history': time_risks
                }
            
        return risk_predictions
    
    def _extrapolate_risk_trend(self, time_risks):
        """推断风险趋势"""
        if len(time_risks) < 2:
            return time_risks[0] if time_risks else 0.5
        
        recent_risks = time_risks[-3:] if len(time_risks) >= 3 else time_risks
        if len(recent_risks) >= 2:
            trend = (recent_risks[-1] - recent_risks[0]) / len(recent_risks)
            predicted = min(max(recent_risks[-1] + trend, 0), 1)
            return predicted
        else:
            return recent_risks[-1]
    
    def _optimize_dynamic_allocation(self, communities, risk_predictions):
        """优化动态资源分配"""
        enhanced_communities = []
        
        for i, community in enumerate(communities):
            community_nodes_list = self._extract_community_nodes(community)
            
            for j, nodes in enumerate(community_nodes_list):
                comm_id = f"{i}_{j}"
                pred = risk_predictions.get(comm_id, {'predicted_risk': 0.5, 'risk_trend': 'stable'})
                
                enhanced_community = {
                    'id': comm_id,
                    'nodes': nodes,
                    'risk_score': pred['predicted_risk'],
                    'size': len(nodes),
                    'risk_trend': pred['risk_trend'],
                    'current_risk': pred['current_risk']
                }
                enhanced_communities.append(enhanced_community)
        
        risk_assessment = {}
        for comm in enhanced_communities:
            for node in comm['nodes']:
                risk_assessment[node] = comm['risk_score'] * 0.8 + np.random.random() * 0.2
        
        resource_constraints = {
            'vaccines': 5000, 
            'test_kits': 3000, 
            'medical_staff': 100, 
            'masks': 10000
        }
        
        allocation_plan = self.optimize_resource_allocation(
            enhanced_communities, risk_assessment, resource_constraints
        )
        
        allocation_plan['dynamics_info'] = {
            'total_communities': len(enhanced_communities),
            'high_risk_count': len([c for c in enhanced_communities if c['risk_score'] >= 0.5]),
            'simulation_steps': len(risk_predictions[list(risk_predictions.keys())[0]]['risk_history']) if risk_predictions else 0
        }
        
        return allocation_plan
    
    def optimize_resource_allocation(self, communities, risk_assessment, resource_constraints):
        """改进版资源分配主方法 - 四级风险分类"""
        allocation_plan = {
            'critical_risk_communities': [],
            'high_risk_communities': [],
            'medium_risk_communities': [],
            'low_risk_communities': [],
            'resource_efficiency': 0.0,
            'estimated_impact': 0.0,
            'risk_distribution': {}
        }
        
        risk_categories = self._categorize_communities_by_risk(communities, risk_assessment)
        
        total_efficiency = 0
        total_communities = 0
        
        for risk_level, comm_list in risk_categories.items():
            for community in comm_list:
                resource_allocation = self._calculate_community_resources(
                    community, risk_level, resource_constraints
                )
                
                efficiency = self._estimate_intervention_efficiency(community, resource_allocation)
                total_efficiency += efficiency
                total_communities += 1
                
                allocation_plan[f'{risk_level}_risk_communities'].append({
                    'community_id': community['id'],
                    'nodes': community['nodes'],
                    'risk_score': community['risk_score'],
                    'size': community['size'],
                    'resource_allocation': resource_allocation,
                    'expected_efficiency': efficiency
                })
        
        if total_communities > 0:
            allocation_plan['resource_efficiency'] = total_efficiency / total_communities
            allocation_plan['estimated_impact'] = self._estimate_overall_impact(allocation_plan)
        
        allocation_plan['risk_distribution'] = {
            level: len(comm_list) for level, comm_list in risk_categories.items()
        }
        
        return allocation_plan
    
    def _categorize_communities_by_risk(self, communities, risk_assessment):
        """改进版风险分类 - 四级分类"""
        risk_categories = {
            'critical': [],
            'high': [],  
            'medium': [],
            'low': []
        }
        
        for community in communities:
            risk_score = community['risk_score']
            
            if risk_score >= 0.7:
                risk_level = 'critical'
            elif risk_score >= 0.5:
                risk_level = 'high'
            elif risk_score >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            risk_categories[risk_level].append(community)
        
        print(f"风险分类统计:")
        for level, comm_list in risk_categories.items():
            avg_risk = np.mean([c['risk_score'] for c in comm_list]) if comm_list else 0
            print(f"   {level}: {len(comm_list)}个社区, 平均风险{avg_risk:.3f}")
        
        return risk_categories
    
    def _calculate_community_resources(self, community, risk_level, constraints):
        """改进版资源分配 - 四级分类"""
        base_allocation = {
            'critical': {
                'vaccines': 150, 'test_kits': 80, 'medical_staff': 8, 'masks': 300
            },
            'high': {
                'vaccines': 100, 'test_kits': 50, 'medical_staff': 5, 'masks': 200
            },
            'medium': {
                'vaccines': 50, 'test_kits': 25, 'medical_staff': 2, 'masks': 100
            },
            'low': {
                'vaccines': 20, 'test_kits': 10, 'medical_staff': 1, 'masks': 50
            }
        }
        
        allocation = base_allocation[risk_level].copy()
        
        size_factor = min(community['size'] / 10, 3.0)
        for resource in allocation:
            allocation[resource] = int(allocation[resource] * size_factor)
        
        return allocation
    
    def _estimate_intervention_efficiency(self, community, resources):
        """估计干预效率"""
        efficiency = 0.0
        
        vaccine_coverage = min(resources['vaccines'] / community['size'], 1.0)
        efficiency += vaccine_coverage * 0.4
        
        testing_capacity = min(resources['test_kits'] / community['size'], 1.0)
        efficiency += testing_capacity * 0.3
        
        medical_support = min(resources['medical_staff'] / max(community['size'] / 100, 1), 1.0)
        efficiency += medical_support * 0.2
        
        protection_supply = min(resources['masks'] / community['size'], 1.0)
        efficiency += protection_supply * 0.1
        
        return min(efficiency, 1.0)
    
    def _estimate_overall_impact(self, allocation_plan):
        """改进版总体影响估计 - 四级权重"""
        impact = 0.0
        weights = {
            'critical': 0.4,
            'high': 0.3,
            'medium': 0.2,  
            'low': 0.1
        }
        
        for risk_level, weight in weights.items():
            for community in allocation_plan[f'{risk_level}_risk_communities']:
                impact += community['expected_efficiency'] * weight
        
        return min(impact, 1.0)