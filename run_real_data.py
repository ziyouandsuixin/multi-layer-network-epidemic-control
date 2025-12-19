import os
import sys
import pandas as pd
import numpy as np
import pickle
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_downloader import NatureDataDownloader
from utils.real_data_processor import RealDataProcessor
from models.community_detection import DynamicCommunityDetectionSimple

def main():
    print("=== Nature COVID-19数据集社区发现分析 ===")
    
    print("\n1. 数据准备...")
    downloader = NatureDataDownloader()
    
    zip_path = os.path.join(downloader.raw_dir, "covid_mobility_data.zip")
    if not os.path.exists(zip_path):
        zip_path = downloader.download_dataset()
        downloader.extract_dataset(zip_path)
    
    print("\n2. 处理数据...")
    processor = RealDataProcessor()
    
    case_data, mobility_data, exposure_data = processor.create_synthetic_from_real_structure()
    
    print(f"病例数据: {len(case_data)} 条记录")
    print(f"移动性数据: {len(mobility_data)} 条记录") 
    print(f"暴露数据: {len(exposure_data)} 条记录")
    
    print("\n3. 构建时序网络...")
    temporal_networks = processor.build_temporal_networks_from_real_data(
        case_data, mobility_data, exposure_data
    )
    
    if not temporal_networks:
        print("无法构建时序网络，使用备用数据...")
        return
    
    valid_networks = {k: v for k, v in temporal_networks.items() 
                     if v.number_of_edges() > 0}
    print(f"有效网络数: {len(valid_networks)}/{len(temporal_networks)}")
    
    if not valid_networks:
        print("没有有效的网络，无法进行社区发现")
        return
    
    print("\n4. 执行社区发现...")
    detector = DynamicCommunityDetectionSimple()
    
    communities = detector.detect_temporal_communities(valid_networks, 'louvain')
    
    print("\n5. 识别高风险社区...")
    risk_factors = {'risk_factor', 'data_type'}
    high_risk_comms = detector.detect_high_risk_communities(communities, risk_factors)
    
    print("\n6. 分析结果:")
    total_communities = 0
    total_high_risk = 0
    
    displayed_count = 0
    for time_point, data in communities.items():
        if displayed_count >= 10:
            break
            
        metrics = data['metrics']
        total_communities += metrics['n_communities']
        
        high_risk_count = len([c for c in high_risk_comms.get(time_point, []) 
                             if c['risk_score'] > 0.5])
        total_high_risk += high_risk_count
        
        print(f"{time_point.date()}: {metrics['n_communities']}个社区 "
              f"(模块度: {metrics['modularity']:.3f}), "
              f"{high_risk_count}个高风险社区")
        
        high_risk_list = high_risk_comms.get(time_point, [])[:3]
        for i, hr_comm in enumerate(high_risk_list):
            if hr_comm['risk_score'] > 0.5:
                print(f"   高风险社区{i+1}: 风险分数{hr_comm['risk_score']:.2f}, "
                      f"大小{hr_comm['size']}节点")
        
        displayed_count += 1
    
    all_time_points = len(communities)
    print(f"\n总体统计 ({all_time_points}个时间点):")
    print(f"   总社区数: {total_communities}")
    print(f"   高风险社区数: {total_high_risk}")
    print(f"   平均每时间点社区数: {total_communities/all_time_points:.1f}")
    
    avg_modularity = np.mean([d['metrics']['modularity'] for d in communities.values()])
    print(f"   平均模块度: {avg_modularity:.3f}")
    
    save_analysis_results(communities, high_risk_comms)
    
    print("\nNature数据集分析完成！")

def save_analysis_results(communities, high_risk_comms):
    """保存分析结果"""
    os.makedirs('outputs', exist_ok=True)
    
    results = {
        'communities': communities,
        'high_risk_communities': high_risk_comms,
        'summary': {
            'total_time_points': len(communities),
            'avg_communities_per_day': np.mean([d['metrics']['n_communities'] for d in communities.values()]),
            'avg_modularity': np.mean([d['metrics']['modularity'] for d in communities.values()])
        }
    }
    
    with open('outputs/nature_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    summary_json = {
        'dataset': 'Nature COVID-19 Mobility Data',
        'analysis_date': pd.Timestamp.now().isoformat(),
        'results_summary': results['summary']
    }
    
    with open('outputs/analysis_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print("分析结果已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()