#!/usr/bin/env python3
"""
优化版公共卫生响应系统
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multilayer_network import MultilayerEpidemicNetwork
from models.dynamic_community_simple import DynamicCommunityDetectionSimple as DynamicCommunityDetection
from models.resource_allocation import ResourceAllocator
from evaluation.performance_evaluator import InterventionEvaluator
from utils.real_data_processor import RealDataProcessor

class OptimizedResourceAllocator(ResourceAllocator):
    """优化版资源分配器"""
    
    def _add_to_risk_category(self, risk_categories, comm_id, nodes, risk_score):
        """优化风险分类阈值"""
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

def main():
    print("=== 优化版公共卫生响应系统 ===")
    
    print("\n1. 数据准备...")
    processor = RealDataProcessor()
    case_data, mobility_data, exposure_data = processor.create_synthetic_from_real_structure()
    
    print("\n2. 构建双层集合种群网络...")
    multilayer_net = MultilayerEpidemicNetwork()
    physical_net = multilayer_net.build_physical_layer(mobility_data, exposure_data)
    information_net = multilayer_net.build_information_layer(physical_net)
    
    print("\n3. 执行动态社区发现...")
    community_detector = DynamicCommunityDetection()
    communities = community_detector.detect_cross_layer_communities(multilayer_net)
    
    print("\n4. 资源分配优化...")
    resource_allocator = OptimizedResourceAllocator()
    
    risk_assessment = {f"person_{i:03d}": np.random.uniform(0.1, 0.9) for i in range(500)}
    resource_constraints = {
        'vaccines': 5000, 'test_kits': 3000, 'medical_staff': 100, 'masks': 10000
    }
    
    allocation_plan = resource_allocator.optimize_resource_allocation(
        communities['physical'], risk_assessment, resource_constraints
    )
    
    print("\n5. 效果评估...")
    evaluator = InterventionEvaluator()
    baseline_plan = {'resource_efficiency': 0.5, 'estimated_impact': 0.4}
    
    evaluation_results = evaluator.evaluate_intervention(allocation_plan, baseline_plan, {})
    
    print("\n6. 优化后系统运行结果:")
    print(f"资源分配效率: {allocation_plan['resource_efficiency']:.3f}")
    print(f"预计防控效果: {allocation_plan['estimated_impact']:.3f}")
    
    print(f"\n风险社区分布:")
    print(f"   高风险社区数: {len(allocation_plan['high_risk_communities'])}")
    print(f"   中风险社区数: {len(allocation_plan['medium_risk_communities'])}") 
    print(f"   低风险社区数: {len(allocation_plan['low_risk_communities'])}")
    
    high_risk_comms = allocation_plan['high_risk_communities'][:3]
    if high_risk_comms:
        print(f"\n最高风险社区:")
        for i, comm in enumerate(high_risk_comms):
            print(f"   社区{comm['id']}: 风险分数{comm['risk_score']:.2f}, 大小{comm['size']}节点")
    
    print("\n优化版系统运行完成！")

if __name__ == "__main__":
    main()