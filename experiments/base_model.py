# experiments/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class EpidemicModel(ABC):
    """所有流行病模型的统一接口"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.results = {}
    
    @abstractmethod
    def simulate(self, network_data: Any, initial_states: Dict, time_steps: int) -> Dict[str, Any]:
        """运行模型模拟
        Args:
            network_data: 网络数据（格式因模型而异）
            initial_states: 统一的初始状态 {infected: [], susceptible: [], ...}
            time_steps: 模拟时间步长
        Returns:
            模拟结果字典
        """
        pass
    
    @abstractmethod
    def allocate_resources(self, risk_assessment: Dict, available_resources: Dict) -> Dict[str, float]:
        """资源分配方法
        Args:
            risk_assessment: 风险评估结果
            available_resources: 可用资源总量
        Returns:
            各社区资源分配比例
        """
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """获取模型评估指标"""
        return self.results.get('metrics', {})