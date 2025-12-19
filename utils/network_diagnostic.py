import networkx as nx
import pandas as pd

def diagnose_temporal_networks(temporal_networks):
    """诊断时序网络"""
    print("=== 时序网络诊断 ===")
    
    stats = []
    for time_point, graph in temporal_networks.items():
        stats.append({
            'date': time_point,
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0,
            'isolated_nodes': len(list(nx.isolates(graph))),
            'components': nx.number_connected_components(graph)
        })
    
    stats_df = pd.DataFrame(stats)
    
    print("网络统计摘要:")
    print(f"总时间点: {len(temporal_networks)}")
    print(f"平均节点数: {stats_df['nodes'].mean():.1f}")
    print(f"平均边数: {stats_df['edges'].mean():.1f}")
    print(f"空网络数: {len(stats_df[stats_df['edges'] == 0])}")
    print(f"孤立节点网络数: {len(stats_df[stats_df['isolated_nodes'] > 0])}")
    
    return stats_df