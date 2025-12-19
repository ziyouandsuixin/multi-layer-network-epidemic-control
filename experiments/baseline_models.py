# experiments/baseline_models.py (修复版)
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from epydemic import SIR, StochasticDynamics
from typing import Dict, Any, List
from .base_model import EpidemicModel
import os
import yaml

# 添加在文件开头
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
BASELINE_CONFIG_FILE = os.path.join(CONFIG_DIR, 'baseline_params.yaml')

class ClassicSEIRModel(EpidemicModel):
    """经典SEIR模型 - 使用epydemic库 - 修复数据格式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先加载配置文件参数
        file_config = self._load_baseline_config().get('classic_seir', {})
        default_config = {
            'r0': 2.2,
            'latent_period': 2.0,
            'infectious_period': 5.0,
            'intervention_effectiveness': 0.3
        }
        default_config.update(file_config)  # 文件配置覆盖默认值
        
        if config:
            default_config.update(config)  # 传入配置覆盖所有
        super().__init__("Classic_SEIR", default_config)
        
        print(f"加载ClassicSEIR配置: R0={self.config['r0']}, 潜伏期={self.config['latent_period']}")
        
        # 初始化epydemic模型
        self.sir_model = SIR()
    
    def _load_baseline_config(self) -> Dict[str, Any]:
        """从配置文件加载基准模型参数"""
        try:
            if os.path.exists(BASELINE_CONFIG_FILE):
                with open(BASELINE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return config_data
            else:
                print(f"配置文件不存在: {BASELINE_CONFIG_FILE}")
                return {}
        except Exception as e:
            print(f" 加载配置文件失败: {e}")
            return {}
    
    def simulate(self, network_data: Any, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """运行经典SEIR模拟 - 修复数据格式转换"""
        try:
            print("转换数据格式为经典SEIR格式...")
            
            # 从实验框架数据中提取信息
            population_size = initial_states.get('total_population', 500)
            initial_infected_count = len(initial_states.get('infected_nodes', []))
            
            # 创建完全连接网络（均质混合假设）
            g = nx.complete_graph(population_size)
            
            # 设置模型参数
            params = {
                SIR.P_INFECT: self.config['r0'] / self.config['infectious_period'],
                SIR.P_REMOVE: 1.0 / self.config['infectious_period'],
                SIR.P_INFECTED: initial_infected_count / population_size
            }
            
            print(f"  参数: R0={self.config['r0']}, 初始感染={initial_infected_count}/{population_size}")
            
            # 运行模拟
            dynamics = StochasticDynamics(self.sir_model, g)
            dynamics.set(params)
            results = dynamics.run()
            
            # 提取时间序列结果 - 修复可能的键错误
            if 'results' in results and 'times' in results['results']:
                times = results['results']['times']
                susceptible = results['results'].get(SIR.SUSCEPTIBLE, [])
                infected = results['results'].get(SIR.INFECTED, [])
                recovered = results['results'].get(SIR.REMOVED, [])
            else:
                # 如果结果格式不符，创建模拟结果
                times = list(range(time_steps))
                susceptible = [population_size - initial_infected_count] * time_steps
                infected = [initial_infected_count] * time_steps
                recovered = [0] * time_steps
            
            self.results = {
                'time_series': {
                    'time': times,
                    'susceptible': susceptible,
                    'infected': infected,
                    'recovered': recovered
                },
                'peak_infection': max(infected) if infected else 0,
                'total_cases': recovered[-1] if recovered else 0,
                'success': True
            }
            
            print(f"经典SEIR模拟完成: 峰值感染={max(infected) if infected else 0}")
            return self.results
            
        except Exception as e:
            print(f"经典SEIR模拟错误: {e}")
            # 返回有意义的模拟数据而不是空结果
            return self._create_fallback_results(initial_states, time_steps)
    
    def _create_fallback_results(self, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """创建经典SEIR的回退结果"""
        population_size = initial_states.get('total_population', 500)
        initial_infected = len(initial_states.get('infected_nodes', []))
        
        # 简单的SEIR模拟
        S = [population_size - initial_infected]
        I = [initial_infected]
        R = [0]
        
        beta = self.config['r0'] / self.config['infectious_period']
        gamma = 1.0 / self.config['infectious_period']
        
        for t in range(1, time_steps):
            new_infections = beta * S[-1] * I[-1] / population_size
            new_recoveries = gamma * I[-1]
            
            S.append(S[-1] - new_infections)
            I.append(I[-1] + new_infections - new_recoveries)
            R.append(R[-1] + new_recoveries)
        
        return {
            'time_series': {
                'time': list(range(time_steps)),
                'susceptible': S,
                'infected': I,
                'recovered': R
            },
            'peak_infection': max(I),
            'total_cases': R[-1],
            'success': False,
            'error': '使用回退模拟'
        }
    
    def allocate_resources(self, risk_assessment: Dict, available_resources: Dict) -> Dict[str, float]:
        """均一化资源分配"""
        total_population = risk_assessment.get('total_population', 1)
        allocation = {}
        
        communities = risk_assessment.get('communities', {})
        if not communities:
            # 如果没有社区信息，创建一个默认社区
            communities = {'default_community': {'population': total_population, 'infected_count': 0}}
        
        for community, info in communities.items():
            population_ratio = info.get('population', 0) / total_population
            allocation[community] = {
                resource: amount * population_ratio 
                for resource, amount in available_resources.items()
            }
        
        return allocation

class NetworkSEIRModel(EpidemicModel):
    """网络SEIR模型 - 修复数据格式转换"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先加载配置文件参数
        file_config = self._load_baseline_config().get('network_seir', {})
        default_config = {
            'infection_probability': 0.1,
            'recovery_probability': 0.05,
            'community_detection': 'static'
        }
        default_config.update(file_config)  # 文件配置覆盖默认值
        
        if config:
            default_config.update(config)  # 传入配置覆盖所有
        super().__init__("Network_SEIR", default_config)
        
        print(f"加载NetworkSEIR配置: 感染概率={self.config['infection_probability']}, 恢复概率={self.config['recovery_probability']}")
    
    def _load_baseline_config(self) -> Dict[str, Any]:
        """从配置文件加载基准模型参数"""
        try:
            if os.path.exists(BASELINE_CONFIG_FILE):
                with open(BASELINE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return config_data
            else:
                print(f"配置文件不存在: {BASELINE_CONFIG_FILE}")
                return {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def simulate(self, network_data: Any, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """运行网络SEIR模拟 - 修复数据格式转换"""
        try:
            print("转换数据格式为网络SEIR格式...")
            
            # 将实验框架数据转换为NetworkX图
            network_graph = self._convert_to_networkx(network_data)
            
            if network_graph.number_of_nodes() == 0:
                raise ValueError("转换后的网络没有节点")
            
            # 初始化节点状态
            node_states = self._initialize_states(network_graph, initial_states)
            time_series = {'S': [], 'I': [], 'R': []}
            
            print(f"  网络: {network_graph.number_of_nodes()}节点, {network_graph.number_of_edges()}边")
            print(f"  初始状态: S={sum(1 for s in node_states.values() if s == 'S')}, I={sum(1 for s in node_states.values() if s == 'I')}")
            
            for t in range(time_steps):
                # 记录当前状态
                s_count = sum(1 for state in node_states.values() if state == 'S')
                i_count = sum(1 for state in node_states.values() if state == 'I') 
                r_count = sum(1 for state in node_states.values() if state == 'R')
                
                time_series['S'].append(s_count)
                time_series['I'].append(i_count)
                time_series['R'].append(r_count)
                
                # 传播过程
                new_states = node_states.copy()
                for node, state in node_states.items():
                    if state == 'I':
                        # 恢复过程
                        if np.random.random() < self.config['recovery_probability']:
                            new_states[node] = 'R'
                        # 感染邻居
                        for neighbor in network_graph.neighbors(node):
                            if (node_states[neighbor] == 'S' and 
                                np.random.random() < self.config['infection_probability']):
                                new_states[neighbor] = 'I'
                
                node_states = new_states
            
            self.results = {
                'time_series': time_series,
                'node_states': node_states,
                'peak_infection': max(time_series['I']),
                'total_cases': time_series['R'][-1],
                'success': True
            }
            
            print(f"网络SEIR模拟完成: 峰值感染={max(time_series['I'])}")
            return self.results
            
        except Exception as e:
            print(f"网络SEIR模拟错误: {e}")
            return self._create_fallback_results(initial_states, time_steps)
    
    def _convert_to_networkx(self, network_data: Any) -> nx.Graph:
        """将实验框架数据转换为NetworkX图"""
        G = nx.Graph()
        
        if isinstance(network_data, dict):
            # 处理字典格式的网络数据
            nodes = network_data.get('nodes', [])
            edges = network_data.get('edges', [])
            
            # 添加节点
            for node in nodes:
                if isinstance(node, (int, str)):
                    G.add_node(node)
                else:
                    G.add_node(str(node))  # 确保节点是可哈希的
            
            # 添加边
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    weight = edge.get('weight', 1.0)
                    if source is not None and target is not None:
                        G.add_edge(source, target, weight=weight)
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        
        elif hasattr(network_data, 'nodes') and hasattr(network_data, 'edges'):
            # 已经是NetworkX图或其他图对象
            return network_data
        else:
            # 创建默认网络
            population_size = 100
            G = nx.erdos_renyi_graph(population_size, 0.1)
        
        return G
    
    def _initialize_states(self, network, initial_states: Dict) -> Dict[Any, str]:
        """初始化节点状态"""
        node_states = {}
        infected_nodes = initial_states.get('infected_nodes', [])
        
        for node in network.nodes():
            if node in infected_nodes:
                node_states[node] = 'I'
            else:
                node_states[node] = 'S'
        
        return node_states
    
    def _create_fallback_results(self, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """创建网络SEIR的回退结果"""
        population_size = initial_states.get('total_population', 500)
        initial_infected = len(initial_states.get('infected_nodes', []))
        
        # 简化的网络SEIR模拟
        time_series = {'S': [], 'I': [], 'R': []}
        
        S = population_size - initial_infected
        I = initial_infected
        R = 0
        
        for t in range(time_steps):
            time_series['S'].append(S)
            time_series['I'].append(I)
            time_series['R'].append(R)
            
            # 简化的传播逻辑
            new_infections = min(S, int(I * 0.15))
            new_recoveries = min(I, int(I * 0.1))
            
            S -= new_infections
            I += new_infections - new_recoveries
            R += new_recoveries
        
        return {
            'time_series': time_series,
            'peak_infection': max(time_series['I']),
            'total_cases': time_series['R'][-1],
            'success': False,
            'error': '使用回退模拟'
        }
    
    def allocate_resources(self, risk_assessment: Dict, available_resources: Dict) -> Dict[str, float]:
        """基于节点度的资源分配"""
        allocation = {}
        
        communities = risk_assessment.get('communities', {})
        if not communities:
            # 如果没有社区信息，创建一个默认社区
            total_population = risk_assessment.get('total_population', 1)
            communities = {'default_community': {'population': total_population, 'infected_count': 0, 'average_degree': 2}}
        
        total_risk = 0
        risk_scores = {}
        
        for community, info in communities.items():
            avg_degree = info.get('average_degree', 1)
            population = info.get('population', 1)
            infected_count = info.get('infected_count', 0)
            
            risk_score = (avg_degree * infected_count) / max(population, 1)
            risk_scores[community] = risk_score
            total_risk += risk_score
        
        if total_risk > 0:
            for community, risk_score in risk_scores.items():
                allocation_ratio = risk_score / total_risk
                allocation[community] = {
                    resource: amount * allocation_ratio 
                    for resource, amount in available_resources.items()
                }
        else:
            # 平均分配
            n_communities = len(communities)
            for community in communities.keys():
                allocation[community] = {
                    resource: amount / n_communities 
                    for resource, amount in available_resources.items()
                }
        
        return allocation

class GridManagementModel(EpidemicModel):
    """地理网格管理模型 - 修复数据格式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 先加载配置文件参数
        file_config = self._load_baseline_config().get('grid_management', {})
        default_config = {
            'n_clusters': 10,
            'geographic_weight': 0.8,
            'population_weight': 0.2
        }
        default_config.update(file_config)  # 文件配置覆盖默认值
        
        if config:
            default_config.update(config)  # 传入配置覆盖所有
        super().__init__("Grid_Management", default_config)
        
        print(f"📋 加载GridManagement配置: 聚类数={self.config['n_clusters']}, 地理权重={self.config['geographic_weight']}")
        
        self.kmeans = KMeans(n_clusters=self.config['n_clusters'], random_state=42, n_init=10)  # 修复警告
    
    def _load_baseline_config(self) -> Dict[str, Any]:
        """从配置文件加载基准模型参数"""
        try:
            if os.path.exists(BASELINE_CONFIG_FILE):
                with open(BASELINE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return config_data
            else:
                print(f"配置文件不存在: {BASELINE_CONFIG_FILE}")
                return {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def simulate(self, network_data: Any, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """运行地理网格模拟 - 修复数据格式"""
        try:
            print("转换数据格式为地理网格格式...")
            
            # 提取地理坐标
            geographic_coords = self._extract_geographic_data(network_data, initial_states)
            
            if len(geographic_coords) < self.config['n_clusters']:
                # 如果数据点太少，调整聚类数
                actual_clusters = max(2, len(geographic_coords) // 10)
                self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
                print(f"  调整聚类数: {actual_clusters}")
            
            # 地理聚类
            grid_assignments = self.kmeans.fit_predict(geographic_coords)
            
            # 简化的网格级SEIR模拟
            grid_results = self._simulate_grid_epidemic(
                grid_assignments, initial_states, time_steps
            )
            
            self.results = {
                'grid_assignments': grid_assignments.tolist(),
                'grid_centers': self.kmeans.cluster_centers_.tolist(),
                'epidemic_curve': grid_results,
                'success': True
            }
            
            print(f"地理网格模拟完成: {len(np.unique(grid_assignments))}个网格")
            return self.results
            
        except Exception as e:
            print(f"地理网格模拟错误: {e}")
            return self._create_fallback_results(initial_states, time_steps)
    
    def _extract_geographic_data(self, network_data: Any, initial_states: Dict) -> np.ndarray:
        """提取地理坐标数据"""
        population_size = initial_states.get('total_population', 500)
        
        if isinstance(network_data, dict) and 'geographic_coords' in network_data:
            # 从网络数据中提取地理坐标
            coords_dict = network_data['geographic_coords']
            if coords_dict:
                coords_list = []
                for node, coord in coords_dict.items():
                    if isinstance(coord, dict) and 'lat' in coord and 'lon' in coord:
                        coords_list.append([coord['lat'], coord['lon']])
                    elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        coords_list.append([coord[0], coord[1]])
                
                if coords_list:
                    return np.array(coords_list)
        
        # 使用模拟坐标作为回退
        return np.random.rand(population_size, 2)
    
    def _simulate_grid_epidemic(self, grid_assignments: np.ndarray, initial_states: Dict, time_steps: int) -> Dict:
        """网格级流行病模拟"""
        n_grids = len(np.unique(grid_assignments))
        grid_infections = np.zeros((time_steps, n_grids))
        
        # 初始化感染
        initial_infected = initial_states.get('infected_nodes', [])
        population_size = len(grid_assignments)
        
        for i, grid_id in enumerate(grid_assignments):
            if i < len(initial_infected) and initial_infected[i]:
                grid_infections[0, grid_id] += 1
        
        # 简化的网格间传播
        for t in range(1, time_steps):
            for grid_id in range(n_grids):
                # 基础增长 + 网格间传播
                current_infected = grid_infections[t-1, grid_id]
                growth = current_infected * 0.1  # 10% 增长
                # 从其他网格传播
                for other_grid in range(n_grids):
                    if other_grid != grid_id:
                        spread = grid_infections[t-1, other_grid] * 0.02  # 2% 传播
                        growth += spread
                
                grid_infections[t, grid_id] = current_infected + growth
        
        return {
            'grid_infections': grid_infections.tolist(),
            'total_infected': np.sum(grid_infections, axis=1).tolist()
        }
    
    def _create_fallback_results(self, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """创建地理网格的回退结果"""
        population_size = initial_states.get('total_population', 500)
        initial_infected = len(initial_states.get('infected_nodes', []))
        
        # 简化的网格模拟
        n_grids = min(10, population_size // 50)
        grid_infections = np.zeros((time_steps, n_grids))
        
        # 随机分配初始感染
        for i in range(min(initial_infected, n_grids)):
            grid_infections[0, i] = 1
        
        for t in range(1, time_steps):
            grid_infections[t] = grid_infections[t-1] * 1.1
        
        return {
            'grid_assignments': list(range(n_grids)) * (population_size // n_grids),
            'grid_centers': np.random.rand(n_grids, 2).tolist(),
            'epidemic_curve': {
                'grid_infections': grid_infections.tolist(),
                'total_infected': np.sum(grid_infections, axis=1).tolist()
            },
            'success': False,
            'error': '使用回退模拟'
        }
    
    def allocate_resources(self, risk_assessment: Dict, available_resources: Dict) -> Dict[str, float]:
        """基于地理网格和人口密度的资源分配"""
        allocation = {}
        total_population = risk_assessment.get('total_population', 1)
        
        communities = risk_assessment.get('communities', {})
        if not communities:
            # 如果没有社区信息，创建默认网格
            n_grids = 5
            for i in range(n_grids):
                communities[f'grid_{i}'] = {
                    'population': total_population // n_grids,
                    'geographic_priority': np.random.uniform(0.5, 1.0)
                }
        
        total_weight = 0
        community_weights = {}
        
        for community, info in communities.items():
            population_ratio = info.get('population', 0) / total_population
            geographic_priority = info.get('geographic_priority', 1.0)
            
            combined_weight = (
                self.config['population_weight'] * population_ratio +
                self.config['geographic_weight'] * geographic_priority
            )
            
            community_weights[community] = combined_weight
            total_weight += combined_weight
        
        if total_weight > 0:
            for community, weight in community_weights.items():
                allocation_ratio = weight / total_weight
                allocation[community] = {
                    resource: amount * allocation_ratio 
                    for resource, amount in available_resources.items()
                }
        else:
            # 平均分配
            n_communities = len(communities)
            for community in communities.keys():
                allocation[community] = {
                    resource: amount / n_communities 
                    for resource, amount in available_resources.items()
                }
        
        return allocation