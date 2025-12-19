#!/usr/bin/env python3
"""
简化版运行脚本 - 社区发现算法演示
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.community_detection import DynamicCommunityDetectionSimple
    from utils.data_loader import TemporalNetworkBuilder, SimpleDataProcessor
    print("成功导入项目模块")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def main():
    print("=== 社区发现算法演示 ===")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")
    
    print("\n1. 生成示例数据...")
    sample_data = generate_sample_data()
    print(f"生成 {len(sample_data)} 条记录")
    
    print("\n2. 构建时序网络...")
    network_builder = TemporalNetworkBuilder()
    
    physical_networks = network_builder.build_physical_network_from_mobility(sample_data)
    print(f"构建物理网络: {len(physical_networks)} 个时间点")
    
    print("\n3. 执行社区发现...")
    detector = DynamicCommunityDetectionSimple()
    
    physical_communities = detector.detect_temporal_communities(physical_networks, 'louvain')
    
    print("\n4. 社区发现结果:")
    for i, (time_point, data) in enumerate(list(physical_communities.items())[:3]):
        metrics = data['metrics']
        print(f"时间点 {i+1}: {metrics['n_communities']} 个社区, 模块度: {metrics['modularity']:.3f}")
    
    print("\n演示完成！社区发现算法运行成功。")

def generate_sample_data():
    """生成示例数据"""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    sample_data = []
    
    for date in dates:
        for i in range(30):
            sample_data.append({
                'individual_id': f"p{i}",
                'date': date,
                'age': np.random.randint(20, 60),
                'gender': np.random.choice(['M', 'F']),
                'location': f"loc{np.random.randint(1, 4)}",
                'exposure_location': f"exp{np.random.randint(1, 3)}",
                'mobility': np.random.choice(['low', 'medium', 'high'])
            })
    
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    main()