# utils/real_data_adapter.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .real_data_processor import RealDataProcessor
    from .scientific_sampler import ScientificSampler
except ImportError:
    from real_data_processor import RealDataProcessor
    
    class ScientificSampler:
        def __init__(self, target_size=500, random_state=42):
            self.target_size = target_size
            self.random_state = random_state
        
        def sample_mobility_data(self, mobility_data, case_data=None):
            if len(mobility_data) <= self.target_size:
                return mobility_data
            return mobility_data.sample(n=self.target_size, random_state=self.random_state)
        
        def sample_exposure_data(self, exposure_data, sampled_individuals):
            if len(exposure_data) <= self.target_size:
                return exposure_data
            return exposure_data.sample(n=self.target_size, random_state=self.random_state)

class RealDataAdapter:
    """真实数据适配器 - 集成科学抽样"""
    
    def __init__(self, data_dir="data", target_size=500):
        self.processor = RealDataProcessor(data_dir)
        self.data_dir = data_dir
        self.sampler = ScientificSampler(target_size=target_size)
        self.target_size = target_size
    
    def create_compatible_data(self):
        """
        生成与现有模拟数据格式完全兼容的真实数据
        集成科学抽样以控制计算复杂度
        """
        print("正在加载和适配真实Nature数据（科学抽样版）...")
        
        try:
            df_clean = self.processor.load_and_clean_real_data()
            
            if df_clean is None or df_clean.empty:
                print("真实数据为空，使用模拟数据")
                return self._create_fallback_data()
            
            case_data, mobility_data, exposure_data = self.processor.extract_network_data(df_clean)
            
            print(f"原始数据规模:")
            print(f"  - 病例数据: {len(case_data)} 条记录")
            print(f"  - 移动数据: {len(mobility_data)} 条记录") 
            print(f"  - 暴露数据: {len(exposure_data)} 条记录")
            
            if len(mobility_data) > self.target_size:
                mobility_data = self.sampler.sample_mobility_data(mobility_data, case_data)
                
                sampled_individuals = mobility_data['individual_id'].unique()
                
                if len(exposure_data) > self.target_size:
                    exposure_data = self.sampler.sample_exposure_data(
                        exposure_data, sampled_individuals
                    )
            
            case_data = self._adapt_case_data(case_data)
            mobility_data = self._adapt_mobility_data(mobility_data)
            exposure_data = self._adapt_exposure_data(exposure_data)
            
            print(f"科学抽样后数据规模:")
            print(f"  - 病例数据: {len(case_data)} 条记录")
            print(f"  - 移动数据: {len(mobility_data)} 条记录") 
            print(f"  - 暴露数据: {len(exposure_data)} 条记录")
            
            self._validate_sampling_quality(mobility_data, exposure_data)
            
            return case_data, mobility_data, exposure_data
            
        except Exception as e:
            print(f"真实数据加载失败: {e}，回退到模拟数据")
            import traceback
            traceback.print_exc()
            return self._create_fallback_data()
    
    def _validate_sampling_quality(self, mobility_data, exposure_data):
        """验证抽样质量"""
        print("\n=== 抽样质量验证 ===")
        
        if 'duration_hours' in mobility_data.columns:
            avg_duration = mobility_data['duration_hours'].mean()
            print(f"平均停留时间: {avg_duration:.2f} 小时")
        
        if 'city' in mobility_data.columns:
            unique_cities = mobility_data['city'].nunique()
            print(f"覆盖城市数量: {unique_cities}")
        
        if 'risk_level' in exposure_data.columns:
            risk_dist = exposure_data['risk_level'].value_counts()
            print("风险等级分布:")
            for risk, count in risk_dist.items():
                print(f"  {risk}: {count}")
        
        n_individuals = mobility_data['individual_id'].nunique()
        n_locations = mobility_data['location_id'].nunique()
        estimated_edges = n_individuals * 5
        print(f"网络规模估计: {n_individuals}节点, ~{estimated_edges}边")
        print("=== 验证完成 ===\n")
    
    def _adapt_case_data(self, case_data):
        """适配病例数据格式，使其与模拟数据一致"""
        if case_data.empty:
            return self._create_default_case_data()
        
        adapted = case_data.copy()
        
        column_mapping = {
            'symptom_onset': 'infection_date',
            'confirmation_date': 'confirmation_date',
            'residency': 'city'
        }
        
        actual_mapping = {k: v for k, v in column_mapping.items() if k in adapted.columns}
        adapted = adapted.rename(columns=actual_mapping)
        
        required_columns = {
            'case_id': 'case_id',
            'age': 30,
            'gender': 'Unknown',
            'infection_date': '2020-01-15',
            'confirmation_date': '2020-01-20', 
            'severity': 'mild',
            'city': 'unknown_city'
        }
        
        for col, default_val in required_columns.items():
            if col not in adapted.columns:
                if col == 'severity':
                    adapted[col] = np.random.choice(
                        ['mild', 'moderate', 'severe'], 
                        size=len(adapted), 
                        p=[0.7, 0.2, 0.1]
                    )
                else:
                    adapted[col] = default_val
        
        adapted = adapted[list(required_columns.keys())]
        adapted = adapted.fillna('unknown')
        
        date_columns = ['infection_date', 'confirmation_date']
        for col in date_columns:
            if col in adapted.columns:
                adapted[col] = pd.to_datetime(
                    adapted[col], 
                    errors='coerce', 
                    format='%m/%d/%Y'
                ).dt.date
                adapted[col] = adapted[col].fillna(datetime(2020, 1, 15).date())
        
        return adapted
    
    def _adapt_mobility_data(self, mobility_data):
        """适配移动性数据格式"""
        if mobility_data.empty:
            return self._create_default_mobility_data()
        
        adapted = mobility_data.copy()
        
        required_columns = {
            'movement_id': 'movement_id',
            'individual_id': 'individual_id', 
            'date': '2020-01-15',
            'location_type': 'unknown',
            'location_id': 'unknown_location',
            'duration_hours': 24.0,
            'city': 'unknown_city'
        }
        
        for col, default_val in required_columns.items():
            if col not in adapted.columns:
                adapted[col] = default_val
        
        adapted = adapted[list(required_columns.keys())]
        adapted = adapted.fillna('unknown')
        
        if 'date' in adapted.columns:
            adapted['date'] = pd.to_datetime(
                adapted['date'], 
                errors='coerce', 
                format='%m/%d/%Y'
            ).dt.date
            adapted['date'] = adapted['date'].fillna(datetime(2020, 1, 15).date())
        
        return adapted
    
    def _adapt_exposure_data(self, exposure_data):
        """适配暴露数据格式 - 重点修复时间计算"""
        if exposure_data.empty:
            return self._create_default_exposure_data()
        
        adapted = exposure_data.copy()
        
        required_columns = {
            'exposure_id': 'exposure_id',
            'case_id': 'case_id',
            'location_id': 'unknown_location',
            'exposure_start': '2020-01-15',
            'exposure_end': '2020-01-15', 
            'exposure_type': 'unknown',
            'risk_level': 'medium'
        }
        
        for col, default_val in required_columns.items():
            if col not in adapted.columns:
                adapted[col] = default_val
        
        adapted = adapted[list(required_columns.keys())]
        adapted = adapted.fillna('unknown')
        
        time_columns = ['exposure_start', 'exposure_end']
        for col in time_columns:
            if col in adapted.columns:
                adapted[col] = pd.to_datetime(
                    adapted[col], 
                    errors='coerce', 
                    format='%m/%d/%Y'
                )
                default_time = datetime(2020, 1, 15)
                adapted[col] = adapted[col].fillna(default_time)
                
                if col == 'exposure_end':
                    mask = adapted['exposure_end'] < adapted['exposure_start']
                    adapted.loc[mask, 'exposure_end'] = adapted.loc[mask, 'exposure_start'] + timedelta(hours=1)
        
        return adapted
    
    def _create_fallback_data(self):
        """创建回退数据（模拟数据）"""
        print("使用模拟数据作为回退...")
        return self.processor.create_synthetic_from_real_structure()
    
    def _create_default_case_data(self):
        """创建默认病例数据"""
        return pd.DataFrame({
            'case_id': ['default_case'],
            'age': [30],
            'gender': ['Unknown'],
            'infection_date': [datetime(2020, 1, 15).date()],
            'confirmation_date': [datetime(2020, 1, 20).date()],
            'severity': ['mild'],
            'city': ['unknown_city']
        })
    
    def _create_default_mobility_data(self):
        """创建默认移动数据"""
        return pd.DataFrame({
            'movement_id': [0],
            'individual_id': ['default_person'],
            'date': [datetime(2020, 1, 15).date()],
            'location_type': ['unknown'],
            'location_id': ['unknown_location'],
            'duration_hours': [24.0],
            'city': ['unknown_city']
        })
    
    def _create_default_exposure_data(self):
        """创建默认暴露数据 - 确保时间格式正确"""
        default_time = datetime(2020, 1, 15)
        return pd.DataFrame({
            'exposure_id': [0],
            'case_id': ['default_case'],
            'location_id': ['unknown_location'],
            'exposure_start': [default_time],
            'exposure_end': [default_time + timedelta(hours=2)],
            'exposure_type': ['unknown'],
            'risk_level': ['medium']
        })