"""
资源约束管理器 - 完善版本
用于管理疫情防控中的资源分配约束
"""

from typing import Dict, Any, List
import yaml
import numpy as np

class ResourceConstraintManager:
    """资源约束管理器 - 完整实现"""
    
    def __init__(self, config_path: str = "experiments/config/resource_constraints.yaml"):
        self.constraints = self._load_constraints(config_path)
        self.priority_rules = self.constraints['priority_rules']
    
    def _load_constraints(self, config_path: str) -> Dict[str, Any]:
        """加载资源约束配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_constraints()
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """默认资源约束配置"""
        return {
            'resource_limits': {
                'testing': {'daily_capacity': 100, 'total_kits': 2000},
                'protective_gear': {'masks': 5000, 'ppe_suits': 1000},
                'medical_resources': {'staff': 50, 'hospital_beds': 80}
            },
            'priority_rules': {
                'level_1': {'conditions': {'min_risk_level': 'critical'}, 'allocation_ratio': 0.40},
                'level_2': {'conditions': {'min_risk_level': 'high', 'min_population_density': 'medium'}, 'allocation_ratio': 0.35},
                'level_3': {'conditions': {'economic_importance': 'high'}, 'allocation_ratio': 0.15},
                'reserve': {'allocation_ratio': 0.10}
            }
        }
    
    def apply_constraints(self, ideal_allocation: Dict[str, Dict], 
                         available_resources: Dict[str, float],
                         community_info: Dict[str, Dict]) -> Dict[str, Dict]:
        """应用资源约束到理想分配方案"""
        
        constrained_allocation = {}
        
        for resource_type, total_available in available_resources.items():
            # 计算该资源类型的总需求
            total_demand = sum(
                allocation.get(resource_type, 0) 
                for allocation in ideal_allocation.values()
            )
            
            if total_demand <= total_available:
                # 需求未超限，保持原分配
                for community, allocation in ideal_allocation.items():
                    if community not in constrained_allocation:
                        constrained_allocation[community] = {}
                    constrained_allocation[community][resource_type] = allocation.get(resource_type, 0)
            else:
                # 需求超限，按优先级缩放
                constrained_allocation = self._scale_allocation_by_priority(
                    ideal_allocation, resource_type, total_available, community_info
                )
        
        return constrained_allocation
    
    def _scale_allocation_by_priority(self, ideal_allocation: Dict, resource_type: str, 
                                    total_available: float, community_info: Dict) -> Dict:
        """按优先级缩放资源分配"""
        
        # 分类社区到优先级等级
        priority_levels = self._classify_communities_by_priority(community_info)
        
        # 计算各等级的资源保障底线
        level_guarantees = self._calculate_level_guarantees(priority_levels, total_available)
        
        # 按优先级分配
        constrained_allocation = {}
        remaining_resources = total_available
        
        for level in ['level_1', 'level_2', 'level_3']:
            level_communities = priority_levels.get(level, [])
            level_guarantee = level_guarantees[level]
            
            # 分配该等级的保障资源
            for community in level_communities:
                if community in ideal_allocation:
                    ideal_amount = ideal_allocation[community].get(resource_type, 0)
                    # 确保至少获得保障额度，但不超理想值
                    allocated = min(ideal_amount, level_guarantee / len(level_communities))
                    
                    if community not in constrained_allocation:
                        constrained_allocation[community] = {}
                    constrained_allocation[community][resource_type] = allocated
                    remaining_resources -= allocated
        
        # 用剩余资源按原比例补充
        if remaining_resources > 0:
            total_remaining_demand = sum(
                ideal_allocation[comm].get(resource_type, 0) - constrained_allocation.get(comm, {}).get(resource_type, 0)
                for comm in ideal_allocation.keys()
            )
            
            if total_remaining_demand > 0:
                scale_factor = remaining_resources / total_remaining_demand
                for community in ideal_allocation:
                    ideal_amount = ideal_allocation[community].get(resource_type, 0)
                    current_amount = constrained_allocation.get(community, {}).get(resource_type, 0)
                    additional = (ideal_amount - current_amount) * scale_factor
                    
                    if community not in constrained_allocation:
                        constrained_allocation[community] = {}
                    constrained_allocation[community][resource_type] = current_amount + additional
        
        return constrained_allocation
    
    def _classify_communities_by_priority(self, community_info: Dict[str, Dict]) -> Dict[str, List]:
        """根据优先级规则分类社区"""
        priority_levels = {'level_1': [], 'level_2': [], 'level_3': []}
        
        for community_id, info in community_info.items():
            risk_level = info.get('risk_level', 'low')
            population_density = info.get('population_density', 'low')
            economic_importance = info.get('economic_importance', 'low')
            
            # 等级1: 极高风险社区
            if risk_level == 'critical':
                priority_levels['level_1'].append(community_id)
            
            # 等级2: 高风险+中等人口密度
            elif risk_level == 'high' and population_density in ['medium', 'high']:
                priority_levels['level_2'].append(community_id)
            
            # 等级3: 经济重要性高
            elif economic_importance == 'high':
                priority_levels['level_3'].append(community_id)
        
        return priority_levels
    
    def _calculate_level_guarantees(self, priority_levels: Dict[str, List], total_resources: float) -> Dict[str, float]:
        """计算各优先级等级的保障资源量"""
        guarantees = {}
        
        for level, rule in self.priority_rules.items():
            if level != 'reserve':
                level_ratio = rule['allocation_ratio']
                n_communities = len(priority_levels.get(level, []))
                
                if n_communities > 0:
                    guarantees[level] = total_resources * level_ratio
                else:
                    guarantees[level] = 0
        
        return guarantees
    
    def validate_allocation(self, allocation: Dict[str, Dict], available_resources: Dict[str, float]) -> bool:
        """验证分配方案是否满足约束"""
        for resource_type, total_available in available_resources.items():
            total_allocated = sum(
                alloc.get(resource_type, 0) for alloc in allocation.values()
            )
            # 允许1%的浮点误差
            if total_allocated > total_available * 1.01:
                return False
        
        return True