import networkx as nx
import community as community_louvain
from collections import defaultdict
import os
import yaml

class ParameterOptimizer:
    """参数优化器 - 独立模块"""
    
    def __init__(self, config_path="configs/optimized_params.yaml"):
        self.config_path = config_path
        self.optimized_params = self._load_optimized_params()
    
    def _load_optimized_params(self):
        """加载优化后的参数"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    params = yaml.safe_load(f)
                    print(f"成功加载优化参数 from {self.config_path}")
                    return params
            else:
                print("优化参数文件不存在，使用默认参数")
                return self._get_default_params()
        except Exception as e:
            print(f"加载优化参数失败，使用默认参数: {e}")
            return self._get_default_params()
    
    def _get_default_params(self):
        """默认参数"""
        return {
            'infection': {
                'base_prob': 0.05,
                'pressure_weight': 0.02,
                'max_infection_prob': 0.3,
            },
            'recovery': {
                'base_rate': 0.15,
                'time_decay': 0.05,
                'min_recovery_prob': 0.1,
                'max_recovery_prob': 0.4,
            },
            'community': {
                'physical_resolution': 0.8,
                'information_resolution': 1.2,
                'min_community_size': 5,
            },
            'simulation': {
                'time_steps': 10,
                'initial_infected_ratio': 0.05,
            }
        }
    
    def get_infection_probability(self, pressure, is_super_spreader=False):
        """获取优化后的感染概率"""
        params = self.optimized_params['infection']
        base_prob = params['base_prob']
        pressure_weight = params['pressure_weight']
        max_prob = params['max_infection_prob']
        
        prob = base_prob + pressure_weight * pressure
        
        if is_super_spreader:
            prob *= 1.5
        
        result = min(prob, max_prob)
        return result
    
    def get_recovery_probability(self, infected_duration):
        """获取优化后的恢复概率"""
        params = self.optimized_params['recovery']
        base_rate = params['base_rate']
        time_decay = params['time_decay']
        min_prob = params['min_recovery_prob']
        max_prob = params['max_recovery_prob']
        
        prob = base_rate + time_decay * infected_duration
        return max(min_prob, min(prob, max_prob))
    
    def get_community_resolution(self, layer_type):
        """获取社区检测分辨率参数"""
        params = self.optimized_params['community']
        if layer_type == 'physical':
            return params.get('physical_resolution', 0.8)
        else:
            return params.get('information_resolution', 1.2)
    
    def get_min_community_size(self):
        """获取最小社区大小"""
        return self.optimized_params['community'].get('min_community_size', 5)
    
    def update_parameters(self, new_params):
        """动态更新参数"""
        self.optimized_params.update(new_params)
        print("参数已更新")
    
    def reload_parameters(self):
        """重新加载参数文件"""
        self.optimized_params = self._load_optimized_params()
    
    def print_current_params(self):
        """打印当前参数"""
        print("当前优化参数:")
        print(f"  感染 - 基础概率: {self.optimized_params['infection']['base_prob']}")
        print(f"  感染 - 压力权重: {self.optimized_params['infection']['pressure_weight']}")
        print(f"  感染 - 最大概率: {self.optimized_params['infection']['max_infection_prob']}")
        print(f"  恢复 - 基础率: {self.optimized_params['recovery']['base_rate']}")
        print(f"  恢复 - 时间衰减: {self.optimized_params['recovery']['time_decay']}")

class CommunityStructure:
    def __init__(self, communities):
        self.communities = communities
        self.community_count = len(communities)

class DynamicCommunityDetectionSimple:
    def __init__(self, use_optimized_params=True):
        self.physical_communities = None
        self.information_communities = None
        
        if use_optimized_params:
            self.param_optimizer = ParameterOptimizer()
            print("参数优化器已启用")
        else:
            self.param_optimizer = None
            print("使用默认参数")
    
    def calculate_optimized_infection_prob(self, pressure, is_super_spreader=False):
        """
        使用优化参数计算感染概率
        
        Args:
            pressure: 感染压力值
            is_super_spreader: 是否为超级传播者
        
        Returns:
            float: 优化后的感染概率
        """
        if self.param_optimizer:
            return self.param_optimizer.get_infection_probability(pressure, is_super_spreader)
        else:
            return min(0.95, 0.03 + 0.03 * pressure)
    
    def calculate_optimized_recovery_prob(self, infected_duration):
        """
        使用优化参数计算恢复概率
        
        Args:
            infected_duration: 感染时长
        
        Returns:
            float: 优化后的恢复概率
        """
        if self.param_optimizer:
            return self.param_optimizer.get_recovery_probability(infected_duration)
        else:
            return min(0.95, 0.05 + 0.03 * infected_duration)
    
    def detect_cross_layer_communities(self, multilayer_net):
        """检测跨层社区 - 修复版"""
        print("开始跨层社区检测...")
        
        communities = {}
        
        try:
            if hasattr(multilayer_net, 'physical_network') and multilayer_net.physical_network is not None:
                print("检测物理层社区...")
                physical_communities = self._detect_layer_communities_simple(
                    multilayer_net, 'physical'
                )
                communities['physical'] = physical_communities
                print(f"物理层发现 {len(physical_communities.communities)} 个社区")
            else:
                print("物理层网络不存在，跳过检测")
                communities['physical'] = CommunityStructure([])
            
            if hasattr(multilayer_net, 'information_network') and multilayer_net.information_network is not None:
                print("检测信息层社区...")
                information_communities = self._detect_layer_communities_simple(
                    multilayer_net, 'information'
                )
                communities['information'] = information_communities
                print(f"信息层发现 {len(information_communities.communities)} 个社区")
            else:
                print("信息层网络不存在，跳过检测")
                communities['information'] = CommunityStructure([])
                
        except Exception as e:
            print(f"社区检测失败: {e}")
            communities = {
                'physical': CommunityStructure([]),
                'information': CommunityStructure([])
            }
        
        return communities

    def _detect_layer_communities_simple(self, multilayer_net, layer_type):
        """修复版：检测单层社区 - 使用优化参数"""
        try:
            print(f"开始检测 {layer_type} 层社区...")
            
            if layer_type == 'physical':
                graph = multilayer_net.physical_network
            elif layer_type == 'information':
                graph = multilayer_net.information_network
            else:
                return CommunityStructure([])
            
            if graph is None or graph.number_of_nodes() == 0:
                return CommunityStructure([])
            
            print(f"   {layer_type} 层网络: {graph.number_of_nodes()}节点, {graph.number_of_edges()}边")
            
            if self.param_optimizer:
                resolution = self.param_optimizer.get_community_resolution(layer_type)
                min_size = self.param_optimizer.get_min_community_size()
            else:
                resolution = 0.8 if layer_type == 'physical' else 1.2
                min_size = 5
                
            print(f"   使用分辨率参数: {resolution}")
            
            partition = community_louvain.best_partition(
                graph, 
                resolution=resolution
            )
            
            communities_dict = defaultdict(list)
            for node, community_id in partition.items():
                communities_dict[community_id].append(node)
            
            communities_list = list(communities_dict.values())
            print(f"   Louvain 初始检测: {len(communities_list)} 个社区")
            
            communities_list = self._merge_small_communities(communities_list, min_size)
            
            print(f"在 {layer_type} 层发现 {len(communities_list)} 个社区")
            
            return CommunityStructure(communities_list)
            
        except Exception as e:
            print(f"{layer_type} 层社区检测失败: {e}")
            return CommunityStructure([])

    def _merge_small_communities(self, communities, min_size=5):
        """合并过小的社区"""
        if not communities:
            return []
        
        small_communities = [comm for comm in communities if len(comm) < min_size]
        large_communities = [comm for comm in communities if len(comm) >= min_size]
        
        print(f"   社区合并前: {len(communities)} 个社区")
        print(f"   小社区(<{min_size}节点): {len(small_communities)} 个")
        print(f"   大社区(>={min_size}节点): {len(large_communities)} 个")
        
        if not large_communities:
            merged_community = []
            for small_comm in small_communities:
                merged_community.extend(small_comm)
            return [merged_community] if merged_community else communities
        
        merged_communities = large_communities.copy()
        
        for small_comm in small_communities:
            merged_communities[0].extend(small_comm)
        
        print(f"   社区合并后: {len(merged_communities)} 个社区")
        
        total_nodes = sum(len(comm) for comm in merged_communities)
        original_nodes = sum(len(comm) for comm in communities)
        print(f"   节点数验证: 合并前{original_nodes} -> 合并后{total_nodes}")
        
        return merged_communities

    def _create_default_communities(self, graph):
        """为空的网络创建默认社区"""
        nodes = list(graph.nodes())
        if not nodes:
            return CommunityStructure([])
        
        if len(nodes) <= 10:
            communities = [[node] for node in nodes]
        else:
            community_size = max(1, len(nodes) // 5)
            communities = []
            for i in range(0, len(nodes), community_size):
                community = nodes[i:i + community_size]
                communities.append(community)
        
        print(f"创建了 {len(communities)} 个默认社区")
        return CommunityStructure(communities)

    def _create_node_based_communities(self, graph):
        """为没有边的网络创建基于节点的社区"""
        nodes = list(graph.nodes())
        if not nodes:
            return CommunityStructure([])
        
        communities = []
        current_community = []
        
        for i, node in enumerate(nodes):
            current_community.append(node)
            if len(current_community) >= 5 or i == len(nodes) - 1:
                communities.append(current_community)
                current_community = []
        
        print(f"创建了 {len(communities)} 个基于节点的社区")
        return CommunityStructure(communities)

    def get_community_overlap(self):
        """获取跨层社区重叠信息"""
        if self.physical_communities is None or self.information_communities is None:
            return {}
        
        overlap_info = {
            'physical_community_count': len(self.physical_communities.communities),
            'information_community_count': len(self.information_communities.communities),
            'overlap_matrix': []
        }
        
        return overlap_info
    
    def reload_optimized_params(self):
        """重新加载优化参数"""
        if self.param_optimizer:
            self.param_optimizer.reload_parameters()
    
    def print_optimized_params(self):
        """打印当前使用的优化参数"""
        if self.param_optimizer:
            self.param_optimizer.print_current_params()
        else:
            print("参数优化器未启用")

def create_parameter_optimizer(config_path="configs/optimized_params.yaml"):
    """创建独立的参数优化器实例"""
    return ParameterOptimizer(config_path)

def get_optimized_infection_prob(pressure, config_path="configs/optimized_params.yaml", 
                               is_super_spreader=False):
    """独立函数：获取优化后的感染概率"""
    optimizer = ParameterOptimizer(config_path)
    return optimizer.get_infection_probability(pressure, is_super_spreader)

def get_optimized_recovery_prob(infected_duration, config_path="configs/optimized_params.yaml"):
    """独立函数：获取优化后的恢复概率"""
    optimizer = ParameterOptimizer(config_path)
    return optimizer.get_recovery_probability(infected_duration)

def test_optimized_parameters():
    """测试优化参数功能"""
    print("测试优化参数功能...")
    
    pressure_test = 10
    infection_prob = get_optimized_infection_prob(pressure_test)
    recovery_prob = get_optimized_recovery_prob(2)
    
    print(f"   压力 {pressure_test} 的感染概率: {infection_prob:.3f}")
    print(f"   感染时长 2 的恢复概率: {recovery_prob:.3f}")
    
    detector = DynamicCommunityDetectionSimple(use_optimized_params=True)
    detector.print_optimized_params()
    
    print("参数优化功能测试完成")

if __name__ == "__main__":
    test_optimized_parameters()