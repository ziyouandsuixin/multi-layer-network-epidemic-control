#!/usr/bin/env python3
"""
完整系统运行：双层网络 + 动态社区发现 + 资源分配 + 效果评估
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multilayer_network import MultilayerEpidemicNetwork
from models.dynamic_community_simple import DynamicCommunityDetectionSimple
from models.resource_allocation import ResourceAllocator
from evaluation.performance_evaluator import InterventionEvaluator
from utils.real_data_processor import RealDataProcessor

def main():
    print("=== 完整公共卫生响应系统 ===")
    
    print("\n1. 数据准备...")
    processor = RealDataProcessor()
    case_data, mobility_data, exposure_data = processor.create_synthetic_from_real_structure()
    
    print("\n2. 构建双层集合种群网络...")
    multilayer_net = MultilayerEpidemicNetwork()
    physical_net = multilayer_net.build_physical_layer(mobility_data, exposure_data)
    information_net = multilayer_net.build_information_layer(physical_net)
    
    print("\n3. 执行动态社区发现...")
    community_detector = DynamicCommunityDetectionSimple()
    communities = community_detector.detect_cross_layer_communities(multilayer_net)
    
    print("\n4. 资源分配优化...")
    resource_allocator = ResourceAllocator()

    risk_assessment = {f"person_{i:03d}": np.random.uniform(0.1, 0.9) for i in range(500)}

    resource_constraints = {
        'vaccines': 5000,
        'test_kits': 3000,
        'medical_staff': 100,
        'masks': 10000
    }

    physical_communities = communities['physical']
    allocation_plan = resource_allocator.optimize_resource_allocation(
        physical_communities, risk_assessment, resource_constraints
    )
    
    print("\n5. 效果评估...")
    evaluator = InterventionEvaluator()
    
    baseline_plan = {
        'resource_efficiency': 0.5,
        'estimated_impact': 0.4,
        'high_risk_communities': [],
        'medium_risk_communities': [],
        'low_risk_communities': []
    }
    
    evaluation_results = evaluator.evaluate_intervention(
        allocation_plan, baseline_plan, {}
    )
    
    print("\n6. 系统运行结果:")
    print(f"资源分配效率: {allocation_plan['resource_efficiency']:.3f}")
    print(f"预计防控效果: {allocation_plan['estimated_impact']:.3f}")
    print(f"综合评估得分: {evaluation_results['overall_score']:.3f}")
    
    print(f"\n详细结果:")
    print(f"   高风险社区数: {len(allocation_plan['high_risk_communities'])}")
    print(f"   中风险社区数: {len(allocation_plan['medium_risk_communities'])}") 
    print(f"   低风险社区数: {len(allocation_plan['low_risk_communities'])}")
    
    print(f"\n疫情控制效果:")
    control_metrics = evaluation_results['epidemic_control']
    print(f"   峰值降低: {control_metrics['peak_reduction']:.1%}")
    print(f"   持续时间缩短: {control_metrics['duration_reduction']:.1%}")
    print(f"   总病例减少: {control_metrics['total_cases_reduction']:.1%}")
    
    print(f"\n与基线方案比较:")
    comparison = evaluation_results['comparative_analysis']
    print(f"   效率提升: {comparison['improvement_in_efficiency']:.3f}")
    print(f"   效果提升: {comparison['improvement_in_impact']:.3f}")
    print(f"   成本节约: {comparison['cost_savings']:.0f} 单位")
    
    print("\n完整系统运行完成！")

if __name__ == "__main__":
    main()