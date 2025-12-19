"""
数据统一器 - 确保所有模型使用相同数据
"""

import numpy as np
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class DataUnifier:
    """数据统一器 - 确保所有模型使用相同数据"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def prepare_unified_data(self, original_data: Any = None, target_size: int = 500) -> Dict[str, Any]:
        """准备统一的实验数据 - 直接复用现有数据加载流程"""
        
        print("准备统一实验数据...")
        
        try:
            from utils.real_data_adapter import RealDataAdapter
            data_adapter = RealDataAdapter(target_size=target_size)
            case_data, mobility_data, exposure_data = data_adapter.create_compatible_data()
            
            initial_conditions = self._prepare_initial_conditions_from_your_data(
                case_data, mobility_data, exposure_data
            )
            
            network_data = self._prepare_network_data_from_mobility(mobility_data)
            
            resource_data = self._prepare_resource_data()
            
            unified_data = {
                'network_data': network_data,
                'initial_conditions': initial_conditions,
                'available_resources': resource_data,
                'raw_data': {
                    'case_data': case_data,
                    'mobility_data': mobility_data,
                    'exposure_data': exposure_data
                },
                'sampling_info': {
                    'target_size': target_size,
                    'random_seed': self.random_seed,
                    'data_source': 'RealDataAdapter'
                }
            }
            
            print(f"统一数据准备完成:")
            print(f"   网络节点: {len(network_data['nodes'])}")
            print(f"   初始感染: {len(initial_conditions['infected_nodes'])}")
            print(f"   总人口: {initial_conditions['total_population']}")
            
            return unified_data
            
        except ImportError as e:
            print(f"导入RealDataAdapter失败: {e}")
            print("使用模拟数据继续实验...")
            return self._create_fallback_data(target_size)
        except Exception as e:
            print(f"数据准备失败: {e}")
            return self._create_fallback_data(target_size)
    
    def _prepare_initial_conditions_from_your_data(self, case_data, mobility_data, exposure_data) -> Dict[str, Any]:
        """基于数据准备统一的初始条件"""
        
        # 从移动数据中提取个体ID作为节点
        if hasattr(mobility_data, 'individual_id'):
            individual_ids = mobility_data['individual_id'].unique().tolist()
        elif 'individual_id' in mobility_data.columns:
            individual_ids = mobility_data['individual_id'].unique().tolist()
        else:
            individual_ids = list(range(len(mobility_data)))
        
        total_population = len(individual_ids)
        
        # 随机选择初始感染者（5%人口）
        n_initial_infected = max(1, int(total_population * 0.05))
        infected_nodes = np.random.choice(
            individual_ids, size=n_initial_infected, replace=False
        ).tolist()
        
        return {
            'total_population': total_population,
            'infected_nodes': infected_nodes,
            'susceptible_nodes': [node for node in individual_ids if node not in infected_nodes],
            'initial_infection_rate': n_initial_infected / total_population,
            'individual_ids': individual_ids
        }
    
    def _prepare_network_data_from_mobility(self, mobility_data) -> Dict[str, Any]:
        """基于移动数据准备网络数据"""
        
        network_data = {
            'nodes': [],
            'edges': [],
            'node_attributes': {},
            'geographic_coords': {},
            'mobility_network': True
        }
        
        # 提取节点（个体）
        if hasattr(mobility_data, 'individual_id'):
            individuals = mobility_data['individual_id'].unique().tolist()
        elif 'individual_id' in mobility_data.columns:
            individuals = mobility_data['individual_id'].unique().tolist()
        else:
            individuals = list(range(len(mobility_data)))
        
        network_data['nodes'] = individuals
        
        # 基于共同位置创建边
        if 'location_id' in mobility_data.columns:
            location_groups = mobility_data.groupby('location_id')
            edge_id = 0
            
            for location, group in location_groups:
                location_individuals = group['individual_id'].unique().tolist()
                
                # 在同一位置的个体间创建连接
                for i in range(len(location_individuals)):
                    for j in range(i + 1, len(location_individuals)):
                        node1 = location_individuals[i]
                        node2 = location_individuals[j]
                        
                        # 创建边
                        edge_key = f"edge_{edge_id}"
                        network_data['edges'].append({
                            'source': node1,
                            'target': node2,
                            'weight': 0.5,
                            'location': location,
                            'type': 'co_location'
                        })
                        edge_id += 1
        
        # 如果没有生成边，创建一些随机连接
        if len(network_data['edges']) == 0 and len(individuals) > 1:
            for i in range(min(10, len(individuals))):
                for j in range(i + 1, min(i + 3, len(individuals))):
                    network_data['edges'].append({
                        'source': individuals[i],
                        'target': individuals[j],
                        'weight': 0.1,
                        'type': 'random_connection'
                    })
        
        # 生成模拟地理坐标
        for node in individuals:
            network_data['geographic_coords'][node] = {
                'lat': np.random.uniform(30, 40),
                'lon': np.random.uniform(110, 120)
            }
            
            # 添加节点属性
            network_data['node_attributes'][node] = {
                'mobility_count': len(mobility_data[mobility_data['individual_id'] == node]) if 'individual_id' in mobility_data.columns else 1,
                'risk_level': np.random.choice(['low', 'medium', 'high'])
            }
        
        return network_data
    
    def _prepare_resource_data(self) -> Dict[str, float]:
        """准备统一的资源数据"""
        return {
            'testing_kits': 2000,
            'masks': 5000,
            'ppe_suits': 1000,
            'medical_staff': 50,
            'hospital_beds': 80
        }
    
    def _create_fallback_data(self, target_size: int) -> Dict[str, Any]:
        """创建回退数据（当主要数据加载失败时）"""
        print("创建回退数据...")
        
        # 生成模拟节点
        nodes = list(range(target_size))
        
        # 初始条件
        n_initial_infected = max(1, int(target_size * 0.05))
        infected_nodes = np.random.choice(nodes, size=n_initial_infected, replace=False).tolist()
        
        initial_conditions = {
            'total_population': target_size,
            'infected_nodes': infected_nodes,
            'susceptible_nodes': [node for node in nodes if node not in infected_nodes],
            'initial_infection_rate': n_initial_infected / target_size
        }
        
        # 网络数据
        network_data = {
            'nodes': nodes,
            'edges': [],
            'node_attributes': {},
            'geographic_coords': {}
        }
        
        # 创建一些随机边
        for i in range(min(100, target_size * 2)):
            source = np.random.randint(0, target_size)
            target = np.random.randint(0, target_size)
            if source != target:
                network_data['edges'].append({
                    'source': source,
                    'target': target,
                    'weight': np.random.uniform(0.1, 1.0),
                    'type': 'random'
                })
        
        # 地理坐标和属性
        for node in nodes:
            network_data['geographic_coords'][node] = {
                'lat': np.random.uniform(30, 40),
                'lon': np.random.uniform(110, 120)
            }
            network_data['node_attributes'][node] = {
                'risk_level': np.random.choice(['low', 'medium', 'high'])
            }
        
        unified_data = {
            'network_data': network_data,
            'initial_conditions': initial_conditions,
            'available_resources': self._prepare_resource_data(),
            'raw_data': {'fallback': True},
            'sampling_info': {
                'target_size': target_size,
                'random_seed': self.random_seed,
                'data_source': 'fallback'
            }
        }
        
        return unified_data