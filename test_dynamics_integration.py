#!/usr/bin/env python3
"""
测试动力学融合
"""
import sys
import os
from datetime import datetime

OUTPUT_DIR = r"D:\whu\coursers\Social\1109\outputs"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.real_data_adapter import RealDataAdapter
    from utils.real_data_processor import RealDataProcessor
except ImportError as e:
    print(f"导入警告: {e}")
    from utils.real_data_processor import RealDataProcessor
    class RealDataAdapter:
        def __init__(self, data_dir="data"):
            self.processor = RealDataProcessor(data_dir)
        
        def create_compatible_data(self):
            print("使用模拟数据...")
            return self.processor.create_synthetic_from_real_structure()

from utils.real_data_processor import RealDataProcessor
from models.multilayer_network import MultilayerEpidemicNetwork
from models.dynamic_test import DynamicCommunityDetectionSimple

def setup_output_directory():
    """设置输出目录和文件名"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dynamics_test_{timestamp}.log"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    return filepath

def test_dynamics_integration():
    """测试动力学模型集成 - 科学抽样版"""

    log_filepath = setup_output_directory()
    original_stdout = sys.stdout

    with open(log_filepath, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("=== 测试动力学模型集成（科学抽样版）===")
        print("基于流行病学原理的科学抽样策略:")
        print("- 高频移动者权重更高")
        print("- 早期病例优先保留") 
        print("- 地理覆盖广泛的个体")
        print("- 高风险暴露重点抽样")
        
        try:
            print(f"\n1. 准备真实数据（科学抽样，目标规模: 500）...")
            
            from utils.real_data_adapter import RealDataAdapter
            data_adapter = RealDataAdapter(target_size=500)
            
            case_data, mobility_data, exposure_data = data_adapter.create_compatible_data()
            
            print("\n2. 构建网络（基于科学抽样数据）...")
            multilayer_net = MultilayerEpidemicNetwork()
            
            print("   构建物理层网络...")
            physical_net = multilayer_net.build_physical_layer(mobility_data, exposure_data)
            if physical_net is None:
                print("物理层网络构建失败")
                return
            
            if not hasattr(physical_net, 'nodes'):
                print("物理层网络缺少 nodes 属性")
                return
                
            if not hasattr(physical_net, 'edges'):
                print("物理层网络缺少 edges 属性")
                return
            
            node_count = len(list(physical_net.nodes()))
            edge_count = len(list(physical_net.edges()))
            
            print(f"   物理层网络: {node_count} 节点, {edge_count} 边")
            
            if node_count == 0:
                print("物理层网络没有节点")
                return
            
            print("   构建信息层网络...")
            information_net = multilayer_net.build_information_layer(physical_net)
            
            if information_net is not None:
                info_node_count = len(list(information_net.nodes()))
                info_edge_count = len(list(information_net.edges()))
                print(f"   信息层网络: {info_node_count} 节点, {info_edge_count} 边")
            else:
                print("信息层网络构建失败，但继续物理层测试")
            
            print("\n3. 社区发现...")
            community_detector = DynamicCommunityDetectionSimple()
            communities = community_detector.detect_cross_layer_communities(multilayer_net)
            
            if not communities:
                print("社区检测返回空结果")
                return
                
            if 'physical' not in communities:
                print("没有找到物理层社区")
                return
            
            physical_communities = communities['physical']
            print(f"   物理层社区数: {len(physical_communities.communities)}")
            
            if len(physical_communities.communities) == 0:
                print("物理层社区为空，创建测试社区")
                all_nodes = list(physical_net.nodes())
                test_communities = [all_nodes[i:i+10] for i in range(0, len(all_nodes), 10)]
                if test_communities:
                    physical_communities = type('obj', (object,), {'communities': test_communities})()
                    print(f"   创建了 {len(test_communities)} 个测试社区")
            
            print("\n4. 测试基础功能...")
            
            try:
                from models.resource_allocator_enhanced import ResourceAllocatorEnhanced
                
                print("   初始化增强版资源分配器...")
                resource_allocator = ResourceAllocatorEnhanced()
                
                all_nodes = []
                for community in physical_communities.communities:
                    all_nodes.extend(community)
                all_nodes = list(set(all_nodes))
                
                print(f"   准备初始化 {len(all_nodes)} 个节点的状态...")
                initial_states = resource_allocator.dynamics_model.initialize_node_states(all_nodes)
                
                infected_count = sum(1 for state in initial_states.values() if state['state'] == 'I')
                print(f"   初始状态: {len(initial_states)}节点, {infected_count}感染者")
                
                print("   执行资源分配...")
                allocation_plan, state_evolution = resource_allocator.optimize_resource_allocation_with_dynamics(
                    [physical_communities],
                    [physical_net],
                    initial_states,
                    time_horizon=3
                )
                
                print("\n5. 测试结果:")
                print(f"   模拟时间步: {len(state_evolution)}")
                print(f"   资源分配效率: {allocation_plan.get('resource_efficiency', 0):.3f}")
                
                final_states = state_evolution[-1]
                final_infected = sum(1 for state in final_states.values() if state['state'] == 'I')
                print(f"   最终感染数: {final_infected}")
                print(f"   感染变化: {final_infected - infected_count:+d}")
                
                print(f"\n   风险社区分布:")
                risk_dist = allocation_plan.get('risk_distribution', {})
                print(f"     危急风险: {risk_dist.get('critical', 0)}")
                print(f"     高风险: {risk_dist.get('high', 0)}") 
                print(f"     中风险: {risk_dist.get('medium', 0)}")
                print(f"     低风险: {risk_dist.get('low', 0)}")
                
                print("\n动力学融合测试成功!")
                
            except ImportError as e:
                print(f"导入增强版模块失败: {e}")
            except Exception as e:
                print(f"动力学测试失败: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"测试过程出错: {e}")
            import traceback
            traceback.print_exc()

    sys.stdout = original_stdout
    print(f"模拟完成，日志已保存到: {log_filepath}")

if __name__ == "__main__":
    test_dynamics_integration()