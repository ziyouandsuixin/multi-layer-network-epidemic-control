"""
模型输出结构调试脚本
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.model_factory import ModelFactory
from experiments.our_model_adapter import OurEnhancedModel

def create_test_data():
    """创建测试数据"""
    import networkx as nx
    
    # 创建小型测试网络
    network = nx.erdos_renyi_graph(50, 0.2)
    
    # 创建初始状态
    all_nodes = list(network.nodes())
    n_initial_infected = 5
    infected_nodes = np.random.choice(all_nodes, size=n_initial_infected, replace=False).tolist()
    
    initial_states = {
        'total_population': 50,
        'infected_nodes': infected_nodes,
        'susceptible_nodes': [node for node in all_nodes if node not in infected_nodes],
        'individual_ids': all_nodes
    }
    
    return {
        'network_data': network,
        'initial_states': initial_states,
        'time_steps': 5
    }

def debug_model_outputs():
    """调试所有模型的输出结构"""
    print("开始调试模型输出结构")
    print("=" * 60)
    
    # 准备测试数据
    test_data = create_test_data()
    print("测试数据准备完成")
    print(f"   网络: {test_data['network_data'].number_of_nodes()}节点")
    print(f"   初始感染: {len(test_data['initial_states']['infected_nodes'])}")
    print(f"   时间步长: {test_data['time_steps']}")
    
    # 初始化所有模型
    models = {}
    
    # 您的模型
    try:
        your_model = OurEnhancedModel()
        models['Our_Enhanced_Model'] = your_model
        print("Our_Enhanced_Model 初始化成功")
    except Exception as e:
        print(f"Our_Enhanced_Model 初始化失败: {e}")
    
    # 传统模型
    baseline_models = ModelFactory.create_all_baselines()
    for name, model in baseline_models.items():
        try:
            models[name] = model
            print(f"{name} 初始化成功")
        except Exception as e:
            print(f"{name} 初始化失败: {e}")
    
    print(f"共初始化 {len(models)} 个模型")
    
    # 测试每个模型的输出结构
    print("\n" + "=" * 60)
    print("测试模型输出结构")
    print("=" * 60)
    
    for model_name, model in models.items():
        print(f"测试 {model_name}...")
        
        try:
            # 运行模拟
            results = model.simulate(
                network_data=test_data['network_data'],
                initial_states=test_data['initial_states'],
                time_steps=test_data['time_steps']
            )
            
            # 分析输出结构
            print(f"   模拟成功")
            print(f"   输出键值: {list(results.keys())}")
            
            # 检查关键字段
            required_fields = ['time_series', 'success']
            for field in required_fields:
                if field in results:
                    print(f"   包含 '{field}' 字段")
                    if field == 'time_series':
                        time_series = results['time_series']
                        print(f"      time_series 键值: {list(time_series.keys())}")
                        # 显示时间序列数据样例
                        for key, value in time_series.items():
                            if isinstance(value, list) and len(value) > 0:
                                print(f"      {key}: 长度={len(value)}, 样例={value[:3]}...")
                else:
                    print(f"   缺少 '{field}' 字段")
            
            # 显示其他重要字段
            other_fields = [key for key in results.keys() if key not in required_fields]
            if other_fields:
                print(f"   其他字段: {other_fields}")
                
            # 显示success状态
            if 'success' in results:
                print(f"   success: {results['success']}")
            
        except Exception as e:
            print(f"   模拟失败: {e}")
    
    # 测试资源分配
    print("\n" + "=" * 60)
    print("测试资源分配")
    print("=" * 60)
    
    available_resources = {
        'vaccines': 20,
        'test_kits': 40,
        'medical_staff': 10
    }
    
    for model_name, model in models.items():
        print(f"测试 {model_name} 资源分配...")
        
        try:
            # 先运行模拟获取风险评估
            results = model.simulate(
                network_data=test_data['network_data'],
                initial_states=test_data['initial_states'],
                time_steps=test_data['time_steps']
            )
            
            # 运行资源分配
            risk_assessment = results.get('risk_assessment', {})
            allocation = model.allocate_resources(
                risk_assessment=risk_assessment,
                available_resources=available_resources
            )
            
            print(f"   资源分配成功")
            print(f"   分配结果键值: {list(allocation.keys())}")
            print(f"   分配详情:")
            for community, resources in list(allocation.items())[:3]:
                print(f"      {community}: {resources}")
            
        except Exception as e:
            print(f"   资源分配失败: {e}")

def main():
    """主调试函数"""
    try:
        debug_model_outputs()
        print("调试完成！")
    except Exception as e:
        print(f"调试失败: {e}")

if __name__ == "__main__":
    main()