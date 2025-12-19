#!/usr/bin/env python3
"""
综合诊断系统 - 排查核心算法问题
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multilayer_network import MultilayerEpidemicNetwork
from models.dynamic_community_simple import DynamicCommunityDetectionSimple
from models.resource_allocation import ResourceAllocator
from utils.real_data_processor import RealDataProcessor

def comprehensive_debug():
    print("=== 综合系统诊断 ===")
    
    print("\n1. 数据准备诊断...")
    processor = RealDataProcessor()
    case_data, mobility_data, exposure_data = processor.create_synthetic_from_real_structure()
    
    print(f"病例数据: {case_data.shape}")
    print(f"移动性数据: {mobility_data.shape}")
    print(f"暴露数据: {exposure_data.shape}")
    
    print("\n2. 双层网络诊断...")
    multilayer_net = MultilayerEpidemicNetwork()
    physical_net = multilayer_net.build_physical_layer(mobility_data, exposure_data)
    information_net = multilayer_net.build_information_layer(physical_net)
    
    debug_multilayer_network(multilayer_net)
    
    print("\n3. 社区发现诊断...")
    community_detector = DynamicCommunityDetectionSimple()
    communities = community_detector.detect_cross_layer_communities(multilayer_net)
    
    debug_community_detection(communities)
    
    print("\n4. 风险评估诊断...")
    risk_assessment = {f"person_{i:03d}": np.random.uniform(0.1, 0.9) for i in range(500)}
    
    debug_risk_data_flow(communities, risk_assessment)
    
    print("\n5. 详细风险计算诊断...")
    resource_allocator = ResourceAllocator()
    physical_comms = communities['physical']
    
    for i, comm in enumerate(physical_comms.communities[:5]):
        community_data = {'nodes': list(comm)}
        risk_score = resource_allocator._calculate_comprehensive_risk(community_data, risk_assessment)
        print(f"社区{i}: 大小={len(comm)}, 风险分数={risk_score:.3f}")
        
        node_risks = [risk_assessment.get(node, 0.1) for node in comm]
        print(f"  节点风险: min={min(node_risks):.3f}, max={max(node_risks):.3f}, mean={np.mean(node_risks):.3f}")

def debug_multilayer_network(multilayer_net):
    """双层网络调试"""
    phys_net = multilayer_net.physical_layer
    info_net = multilayer_net.information_layer
    
    print(f"物理层: {phys_net.number_of_nodes()}节点, {phys_net.number_of_edges()}边")
    print(f"信息层: {info_net.number_of_nodes()}节点, {info_net.number_of_edges()}边")
    
    sample_nodes = list(phys_net.nodes())[:3]
    for node in sample_nodes:
        attrs = phys_net.nodes[node]
        print(f"  节点{node}属性: {attrs}")

def debug_community_detection(communities):
    """社区发现调试"""
    physical_comms = communities['physical']
    print(f"物理层社区数: {len(physical_comms.communities)}")
    
    community_sizes = [len(comm) for comm in physical_comms.communities]
    size_stats = pd.Series(community_sizes).describe()
    print(f"社区规模统计:")
    print(f"  最小: {size_stats['min']}, 最大: {size_stats['max']}, 平均: {size_stats['mean']:.1f}")
    
    size_ranges = {'微型(<5)': 0, '小型(5-10)': 0, '中型(10-20)': 0, '大型(>20)': 0}
    for size in community_sizes:
        if size < 5: size_ranges['微型(<5)'] += 1
        elif size < 10: size_ranges['小型(5-10)'] += 1
        elif size < 20: size_ranges['中型(10-20)'] += 1
        else: size_ranges['大型(>20)'] += 1
    
    print(f"社区规模分布: {size_ranges}")

def debug_risk_data_flow(communities, risk_assessment):
    """风险评估数据流调试"""
    physical_comms = communities['physical']
    
    all_community_nodes = set()
    for comm in physical_comms.communities:
        all_community_nodes.update(comm)
    
    coverage = len(set(risk_assessment.keys()) & all_community_nodes) / len(all_community_nodes)
    print(f"风险评估节点覆盖率: {coverage:.1%}")
    
    risk_values = list(risk_assessment.values())
    print(f"风险值范围: {min(risk_values):.3f} - {max(risk_values):.3f}")

if __name__ == "__main__":
    comprehensive_debug()