#!/usr/bin/env python3
"""
测试动力学融合 - 修复版
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.real_data_processor import RealDataProcessor
from models.multilayer_network import MultilayerEpidemicNetwork
from models.dynamic_community_simple import DynamicCommunityDetectionSimple

def test_dynamics_integration():
    """测试动力学模型集成 - 修复版"""
    print("=== 测试动力学模型集成 ===")
    
    try:
        # 1. 准备数据
        print("\n1. 准备测试数据...")
        processor = RealDataProcessor()
        case_data, mobility_data, exposure_data = processor.create_synthetic_from_real_structure()
        
        # 2. 构建网络
        print("\n2. 构建网络...")
        multilayer_net = MultilayerEpidemicNetwork()
        physical_net = multilayer_net.build_physical_layer(mobility_data, exposure_data)
        
        if physical_net is None or physical_net.number_of_nodes() == 0:
            print("物理层网络构建失败")
            return
        
        print(f"   物理层网络: {physical_net.number_of_nodes()}节点, {physical_net.number_of_edges()}边")
        
        # 3. 社区发现
        print("\n3. 社区发现...")
        community_detector = DynamicCommunityDetectionSimple()
        communities = community_detector.detect_cross_layer_communities(multilayer_net)
        
        if not communities or 'physical' not in communities:
            print("社区检测失败")
            return
            
        physical_communities = communities['physical']
        print(f"   发现物理层社区: {len(physical_communities.communities)}个")
        
        # 4. 导入增强版资源分配器
        print("\n4. 初始化动力学模型...")
        try:
            from models.resource_allocator_enhanced import ResourceAllocatorEnhanced, CommunityEpidemicDynamics
            
            dynamics_model = CommunityEpidemicDynamics()
            
            # 初始化状态
            all_nodes = []
            for community in physical_communities.communities:
                all_nodes.extend(community)
            all_nodes = list(set(all_nodes))
            
            initial_states = dynamics_model.initialize_node_states(all_nodes)
            
            infected_count = sum(1 for state in initial_states.values() if state['state'] == 'I')
            print(f"   初始状态: {len(initial_states)}节点, {infected_count}感染者")
            
            # 5. 测试增强版资源分配
            print("\n5. 测试增强版资源分配...")
            resource_allocator = ResourceAllocatorEnhanced()
            
            allocation_plan, state_evolution = resource_allocator.optimize_resource_allocation_with_dynamics(
                [physical_communities],  # 社区列表
                [physical_net],          # 接触网络列表
                initial_states,          # 初始状态
                time_horizon=5           # 预测5个时间步
            )
            
            # 6. 显示结果
            print("\n6. 动力学增强结果:")
            print(f"   模拟时间步: {len(state_evolution)}")
            print(f"   资源分配效率: {allocation_plan.get('resource_efficiency', 0):.3f}")
            
            final_states = state_evolution[-1]
            final_infected = sum(1 for state in final_states.values() if state['state'] == 'I')
            print(f"   最终感染数: {final_infected}")
            print(f"   感染变化: {final_infected - infected_count:+d}")
            
            print(f"\n   风险社区分布:")
            print(f"     高风险社区: {len(allocation_plan.get('high_risk_communities', []))}")
            print(f"     中风险社区: {len(allocation_plan.get('medium_risk_communities', []))}")
            print(f"     低风险社区: {len(allocation_plan.get('low_risk_communities', []))}")
            
            if 'dynamics_info' in allocation_plan:
                dyn_info = allocation_plan['dynamics_info']
                print(f"\n   动力学信息:")
                print(f"     总社区数: {dyn_info['total_communities']}")
                print(f"     高风险社区数: {dyn_info['high_risk_count']}")
                print(f"     模拟步数: {dyn_info['simulation_steps']}")
            
            print("\n动力学融合测试完成!")
            
        except ImportError as e:
            print(f"导入增强版模块失败: {e}")
            return
        except Exception as e:
            print(f"动力学测试失败: {e}")
            import traceback
            traceback.print_exc()
            return
            
    except Exception as e:
        print(f"测试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamics_integration()