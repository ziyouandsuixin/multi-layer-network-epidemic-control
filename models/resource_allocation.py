import numpy as np
import pandas as pd
from collections import defaultdict

class ResourceAllocator:
    """资源分配优化器"""
    
    def __init__(self):
        self.resource_types = {
            'medical': ['vaccines', 'test_kits', 'medical_staff'],
            'non_medical': ['masks', 'sanitizers', 'information_materials'],
            'intervention': ['lockdown', 'travel_restrictions', 'testing_campaigns']
        }
    
    def optimize_resource_allocation(self, communities, risk_assessment, resource_constraints):
        """优化资源分配"""
        print("优化资源分配...")
        
        allocation_plan = {
            'high_risk_communities': [],
            'medium_risk_communities': [],
            'low_risk_communities': [],
            'resource_efficiency': 0.0,
            'estimated_impact': 0.0
        }
        
        risk_categories = self._categorize_communities_by_risk(communities, risk_assessment)
        
        for risk_level, comm_list in risk_categories.items():
            for community in comm_list:
                resource_allocation = self._calculate_community_resources(
                    community, risk_level, resource_constraints
                )
                
                allocation_plan[f'{risk_level}_risk_communities'].append({
                    'community_id': community['id'],
                    'nodes': community['nodes'],
                    'risk_score': community['risk_score'],
                    'resource_allocation': resource_allocation,
                    'expected_efficiency': self._estimate_intervention_efficiency(community, resource_allocation)
                })
        
        allocation_plan['resource_efficiency'] = self._calculate_overall_efficiency(allocation_plan)
        allocation_plan['estimated_impact'] = self._estimate_overall_impact(allocation_plan)
        
        return allocation_plan
    
    def _categorize_communities_by_risk(self, communities, risk_assessment):
        """按风险等级分类社区 - 修复版本"""
        risk_categories = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        if hasattr(communities, 'communities'):
            comm_list = communities.communities
            for comm_id, community_nodes in enumerate(comm_list):
                community = {'nodes': community_nodes}
                risk_score = self._calculate_comprehensive_risk(community, risk_assessment)
                self._add_to_risk_category(risk_categories, comm_id, community_nodes, risk_score)
        
        elif isinstance(communities, list):
            for comm_id, community_nodes in enumerate(communities):
                if hasattr(community_nodes, '__iter__'):
                    community = {'nodes': list(community_nodes)}
                    risk_score = self._calculate_comprehensive_risk(community, risk_assessment)
                    self._add_to_risk_category(risk_categories, comm_id, list(community_nodes), risk_score)
        
        elif isinstance(communities, dict):
            for comm_id, community in communities.items():
                if isinstance(community, dict) and 'nodes' in community:
                    nodes = community['nodes']
                else:
                    nodes = list(community) if hasattr(community, '__iter__') else []
                
                risk_score = self._calculate_comprehensive_risk({'nodes': nodes}, risk_assessment)
                self._add_to_risk_category(risk_categories, comm_id, nodes, risk_score)
        
        else:
            print("警告: 未知的社区数据结构，使用空分类")
        
        return risk_categories

    def _add_to_risk_category(self, risk_categories, comm_id, nodes, risk_score):
        """添加社区到风险分类"""
        if risk_score >= 0.5:
            risk_level = 'high'
        elif risk_score >= 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_categories[risk_level].append({
            'id': comm_id,
            'nodes': nodes,
            'risk_score': risk_score,
            'size': len(nodes)
        })
    
    def _calculate_comprehensive_risk(self, community, risk_assessment):
        """计算综合风险分数"""
        risk_score = 0.0
        
        if community.get('has_superspreader', False):
            risk_score += 0.3
        
        if community.get('high_mobility', False):
            risk_score += 0.2
        
        if community.get('low_awareness', False):
            risk_score += 0.2
        
        node_risk_sum = sum(risk_assessment.get(node, 0.1) for node in community['nodes'])
        avg_node_risk = node_risk_sum / len(community['nodes']) if community['nodes'] else 0
        
        risk_score += avg_node_risk * 0.3
        
        return min(risk_score, 1.0)
    
    def _calculate_community_resources(self, community, risk_level, constraints):
        """计算社区资源分配"""
        base_allocation = {
            'high': {'vaccines': 100, 'test_kits': 50, 'medical_staff': 5, 'masks': 200},
            'medium': {'vaccines': 50, 'test_kits': 25, 'medical_staff': 2, 'masks': 100},
            'low': {'vaccines': 20, 'test_kits': 10, 'medical_staff': 1, 'masks': 50}
        }
        
        allocation = base_allocation[risk_level].copy()
        
        size_factor = min(community['size'] / 10, 2.0)
        for resource in allocation:
            allocation[resource] = int(allocation[resource] * size_factor)
        
        for resource, amount in allocation.items():
            if resource in constraints:
                allocation[resource] = min(amount, constraints[resource])
        
        return allocation
    
    def _estimate_intervention_efficiency(self, community, resources):
        """估计干预效率"""
        efficiency = 0.0
        
        vaccine_coverage = min(resources['vaccines'] / community['size'], 1.0)
        efficiency += vaccine_coverage * 0.4
        
        testing_capacity = min(resources['test_kits'] / community['size'], 1.0)
        efficiency += testing_capacity * 0.3
        
        medical_support = min(resources['medical_staff'] / (community['size'] / 100), 1.0)
        efficiency += medical_support * 0.2
        
        protection_supply = min(resources['masks'] / community['size'], 1.0)
        efficiency += protection_supply * 0.1
        
        return min(efficiency, 1.0)
    
    def _calculate_overall_efficiency(self, allocation_plan):
        """计算总体资源效率"""
        total_efficiency = 0.0
        total_communities = 0
        
        for risk_level in ['high', 'medium', 'low']:
            communities = allocation_plan[f'{risk_level}_risk_communities']
            for community in communities:
                total_efficiency += community['expected_efficiency']
                total_communities += 1
        
        return total_efficiency / total_communities if total_communities > 0 else 0
    
    def _estimate_overall_impact(self, allocation_plan):
        """估计总体干预效果"""
        impact = 0.0
        
        high_risk_weight = 0.5
        medium_risk_weight = 0.3
        low_risk_weight = 0.2
        
        for community in allocation_plan['high_risk_communities']:
            impact += community['expected_efficiency'] * high_risk_weight
        
        for community in allocation_plan['medium_risk_communities']:
            impact += community['expected_efficiency'] * medium_risk_weight
        
        for community in allocation_plan['low_risk_communities']:
            impact += community['expected_efficiency'] * low_risk_weight
        
        return min(impact, 1.0)