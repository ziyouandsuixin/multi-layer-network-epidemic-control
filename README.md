# 多层网络社区发现驱动的传染病精准防控系统研究代码

本代码仓库为论文《面向传染病精准防控的多层网络社区发现与资源优化方法研究》的实现代码。

## 项目概述

研究基于Nature公开的COVID-19多源异构数据，构建物理接触层与信息传播层耦合的双层网络模型。通过参数化调节的Louvain社区发现算法及后处理合并策略，在双层网络上识别具有流行病学管理意义的动态风险社区。结合扩展SIS传播动力学模型，建立四级（危急、高风险、中风险、低风险）社区风险评估与分类机制，开发风险驱动的动态资源分配器。

## 项目结构

```
epidemic-network-community-detection/
│   check.py
│   debug_system.py
│   project_structure.txt
│   README.md
│   run_complete_system.py
│   run_optimized_system.py
│   run_real_data.py
│   run_simple.py
│   test_dynamics_integration.py
│   
├───configs
│       optimized_params.yaml
│       tuning_rules.yaml
│       
├───data
│   │   data_test.py
│   │   
│   ├───processed
│   │   │   synthetic_case_data.csv
│   │   │   synthetic_exposure_data.csv
│   │   │   synthetic_mobility_data.csv
│   │   │   
│   │   └───temporal_networks
│   └───raw
│           covid_mobility_data.zip
│           dataset_EN.csv
│           data_sources.csv
│           reasons_for_missing_data.csv
│           
├───evaluation
│   │   data_extractor.py
│   │   evaluator.py
│   │   metrics_calculator.py
│   │   performance_evaluator.py
│   │   test_metrics.py
│           
├───experiments
│   │   baseline_models.py
│   │   base_model.py
│   │   comparative_experiment.py
│   │   comprehensive_validation_experiment.py.py
│   │   constraint_manager.py
│   │   data_unifier.py
│   │   debug_models.py
│   │   enhanced_evaluator.py
│   │   experiment_runner.py
│   │   main_experiment.py
│   │   model_factory.py
│   │   our_model_adapter.py
│   │   test1113.py
│   │   test1114.py
│   
├───models
│   │   community_detection.py
│   │   dynamic_community.py
│   │   dynamic_community_simple.py
│   │   dynamic_test.py
│   │   epidemic_dynamics.py
│   │   multilayer_network.py
│   │   resource_allocation.py
│   │   resource_allocator_enhanced.py
│   
├───outputs
│       
├───results
├───scripts
│       run_optimization.py
│       
├───utils
│   │   data_downloader.py
│   │   data_loader.py
│   │   network_diagnostic.py
│   │   real_data_adapter.py
│   │   real_data_processor.py
│   │   scientific_sampler.py
│   
└───validation_results
```

## 使用说明

### 环境配置

1）安装依赖包：
```bash
pip install -r requirements.txt
```

2）建议使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 运行程序

1）运行完整系统：
```bash
python run_complete_system.py
```

2）运行优化系统：
```bash
python run_optimized_system.py
```

3）运行主测试程序：
```bash
python test_dynamics_integration.py
```

### 配置文件说明

系统参数配置在configs目录下：

1）optimized_params.yaml：优化后的系统参数
2）tuning_rules.yaml：参数调优规则

### 数据配置

1）数据规模调整：在real_data_adapter.py中修改target_size参数：
```python
def __init__(self, data_dir="data", target_size=500):  # 可调整为其他数值
```

2）数据处理：原始数据应放置在data/raw目录下，处理后的数据将生成在data/processed目录。

## 核心功能模块

### 数据处理
1）real_data_adapter.py：Nature COVID-19真实数据适配器
2）scientific_sampler.py：科学抽样器，控制数据规模
3）data_loader.py：统一数据加载接口

### 模型实现
1）multilayer_network.py：双层网络构建模型
2）dynamic_community.py：动态社区发现算法
3）epidemic_dynamics.py：流行病动力学模拟
4）resource_allocator_enhanced.py：增强版资源分配器

### 评估验证
1）test_metrics.py：主要评估脚本，计算NDCG等指标
2）performance_evaluator.py：系统性能评估器
3）comparative_experiment.py：对比实验模块

## 输出说明

1）程序运行日志保存在outputs目录
2）评估结果保存在validation_results目录
3）实验结果保存在results目录

## 注意事项

1）数据集文件已从仓库中删除，仅保留文件夹结构
2）原始数据需从Nature COVID-19数据集获取
3）中间处理数据需通过运行程序生成
4）输出目录内容为程序运行后自动生成