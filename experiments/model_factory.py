"""
模型工厂类
用于创建各种基线模型
"""

from typing import Dict, Any
import yaml
import os
from .baseline_models import ClassicSEIRModel, NetworkSEIRModel, GridManagementModel
from .base_model import EpidemicModel

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type: str, config_path: str = None) -> EpidemicModel:
        """创建指定类型的模型"""
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
        else:
            configs = {}
        
        model_configs = configs.get('baseline_models', {})
        
        if model_type == "classic_seir":
            config = model_configs.get('classic_seir', {})
            return ClassicSEIRModel(config)
            
        elif model_type == "network_seir":
            config = model_configs.get('network_seir', {})
            return NetworkSEIRModel(config)
            
        elif model_type == "grid_management":
            config = model_configs.get('grid_management', {})
            return GridManagementModel(config)
            
        else:
            raise ValueError(f"未知模型类型: {model_type}")
    
    @staticmethod
    def create_all_baselines(config_path: str = None) -> Dict[str, EpidemicModel]:
        """创建所有基线模型"""
        models = {}
        
        for model_type in ["classic_seir", "network_seir", "grid_management"]:
            try:
                models[model_type] = ModelFactory.create_model(model_type, config_path)
            except Exception as e:
                print(f"创建模型 {model_type} 失败: {e}")
        
        return models