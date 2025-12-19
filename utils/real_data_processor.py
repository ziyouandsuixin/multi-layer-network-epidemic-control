import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from tqdm import tqdm
import os

class RealDataProcessor:
    """真实数据处理和网络构建"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
    
    def load_and_clean_real_data(self):
        """加载并清理真实Nature数据"""
        data_path = os.path.join(self.raw_dir, "dataset_EN.csv")
        
        if not os.path.exists(data_path):
            print(f"真实数据文件不存在: {data_path}")
            return self.create_synthetic_from_real_structure()[0]
        
        print("正在加载真实Nature数据集...")
        df = pd.read_csv(data_path, low_memory=False)
        
        useful_columns = df.columns[:25].tolist()
        df_clean = df[useful_columns].copy()
        
        print(f"清理后数据形状: {df_clean.shape}")
        return df_clean
    
    def extract_network_data(self, df_clean):
        """从清理的数据中提取网络构建所需信息"""
        
        case_data = self._extract_case_data(df_clean)
        
        mobility_data = self._extract_mobility_data(df_clean)
        
        exposure_data = self._extract_exposure_data(df_clean)
        
        return case_data, mobility_data, exposure_data
    
    def _extract_case_data(self, df):
        """提取病例数据"""
        case_data = []
        
        for _, row in df.iterrows():
            if pd.isna(row['ID']):
                continue
                
            case_info = {
                'case_id': row['ID'],
                'age': self._safe_get(row, 'Age'),
                'gender': self._safe_get(row, 'Gender'),
                'occupation': self._safe_get(row, 'Occupation'),
                'residency': self._safe_get(row, 'Place of Residency'),
                'symptom_onset': self._safe_get(row, 'Date_Symptom_Onset'),
                'confirmation_date': self._safe_get(row, 'Date_Confirmation'),
                'hospitalization_date': self._safe_get(row, 'Date_Hospitalisation'),
                'quarantine_date': self._safe_get(row, 'Date_Quarantine')
            }
            case_data.append(case_info)
        
        return pd.DataFrame(case_data)
    
    def _extract_mobility_data(self, df):
        """提取移动性数据"""
        mobility_data = []
        movement_id = 0
        
        for _, row in df.iterrows():
            if pd.isna(row['ID']):
                continue
            
            individual_id = row['ID']
            
            if pd.notna(row['Arrival Date']) and pd.notna(row['Place of Destination']):
                mobility_record = {
                    'movement_id': movement_id,
                    'individual_id': individual_id,
                    'date': row['Arrival Date'],
                    'location_type': 'destination',
                    'location_id': row['Place of Destination'],
                    'duration_hours': 24.0,
                    'city': self._extract_city(row['Place of Destination'])
                }
                mobility_data.append(mobility_record)
                movement_id += 1
            
            if pd.notna(row['Place of Departure']):
                date_to_use = self._get_date_for_movement(row)
                mobility_record = {
                    'movement_id': movement_id,
                    'individual_id': individual_id,
                    'date': date_to_use,
                    'location_type': 'departure', 
                    'location_id': row['Place of Departure'],
                    'duration_hours': 24.0,
                    'city': self._extract_city(row['Place of Departure'])
                }
                mobility_data.append(mobility_record)
                movement_id += 1
            
            if pd.notna(row['Place of Residency']):
                date_to_use = self._get_date_for_movement(row)
                mobility_record = {
                    'movement_id': movement_id,
                    'individual_id': individual_id,
                    'date': date_to_use,
                    'location_type': 'residency',
                    'location_id': row['Place of Residency'],
                    'duration_hours': 24.0,
                    'city': self._extract_city(row['Place of Residency'])
                }
                mobility_data.append(mobility_record)
                movement_id += 1
        
        return pd.DataFrame(mobility_data)
    
    def _extract_exposure_data(self, df):
        """提取暴露数据"""
        exposure_data = []
        exposure_id = 0
        
        for _, row in df.iterrows():
            if pd.isna(row['ID']):
                continue
            
            if pd.notna(row['Contact_ID_Relationship']):
                exposure_record = {
                    'exposure_id': exposure_id,
                    'case_id': row['ID'],
                    'location_id': self._safe_get(row, 'Place and Event', 'unknown_location'),
                    'exposure_start': self._get_exposure_date(row),
                    'exposure_end': self._get_exposure_end_date(row),
                    'exposure_type': 'close_contact',
                    'risk_level': self._assess_risk_level(row['Contact_ID_Relationship'])
                }
                exposure_data.append(exposure_record)
                exposure_id += 1
            
            if pd.notna(row['Venue']):
                exposure_record = {
                    'exposure_id': exposure_id,
                    'case_id': row['ID'],
                    'location_id': row['Venue'],
                    'exposure_start': self._get_exposure_date(row),
                    'exposure_end': self._get_exposure_end_date(row),
                    'exposure_type': 'venue_exposure',
                    'risk_level': 'medium'
                }
                exposure_data.append(exposure_record)
                exposure_id += 1
        
        return pd.DataFrame(exposure_data)
    
    def _safe_get(self, row, column, default='unknown'):
        """安全获取数据"""
        if pd.isna(row[column]):
            return default
        return row[column]
    
    def _extract_city(self, location_str):
        """从地点字符串中提取城市名"""
        if pd.isna(location_str) or location_str == 'unknown':
            return 'unknown_city'
        
        if '_' in str(location_str):
            return str(location_str).split('_')[0]
        else:
            return str(location_str)
    
    def _get_exposure_date(self, row):
        """获取暴露日期 - 返回datetime对象"""
        date_str = None
        if pd.notna(row['Earliest Possible Date']):
            date_str = row['Earliest Possible Date']
        elif pd.notna(row['Arrival Date']):
            date_str = row['Arrival Date']
        else:
            date_str = '2020-01-15'
        
        try:
            return pd.to_datetime(date_str, errors='coerce', format='%m/%d/%Y')
        except:
            return datetime(2020, 1, 15)

    def _get_date_for_movement(self, row):
        """为移动记录获取日期"""
        date_str = None
        if pd.notna(row['Arrival Date']):
            date_str = row['Arrival Date']
        elif pd.notna(row['Earliest Possible Date']):
            date_str = row['Earliest Possible Date']
        else:
            date_str = '2020-01-15'
        
        try:
            return pd.to_datetime(date_str, errors='coerce', format='%m/%d/%Y').strftime('%m/%d/%Y')
        except:
            return '2020-01-15'
    
    def _get_exposure_end_date(self, row):
        """获取暴露结束日期 - 返回datetime对象"""
        start_date = self._get_exposure_date(row)
        return start_date + timedelta(hours=np.random.randint(1, 8))
    
    def _assess_risk_level(self, contact_relationship):
        """根据接触关系评估风险等级"""
        high_risk_keywords = ['family', 'close', 'household']
        medium_risk_keywords = ['colleague', 'work', 'friend']
        
        contact_str = str(contact_relationship).lower()
        
        for keyword in high_risk_keywords:
            if keyword in contact_str:
                return 'high'
        
        for keyword in medium_risk_keywords:
            if keyword in contact_str:
                return 'medium'
        
        return 'low'
    
    def create_synthetic_from_real_structure(self):
        """
        基于Nature数据集结构创建合成数据
        """
        print("基于Nature数据集结构创建合成数据...")
        
        case_data = self._create_synthetic_case_data()
        
        mobility_data = self._create_synthetic_mobility_data()
        
        exposure_data = self._create_synthetic_exposure_data()
        
        case_data.to_csv(os.path.join(self.processed_dir, "synthetic_case_data.csv"), index=False)
        mobility_data.to_csv(os.path.join(self.processed_dir, "synthetic_mobility_data.csv"), index=False)
        exposure_data.to_csv(os.path.join(self.processed_dir, "synthetic_exposure_data.csv"), index=False)
        
        print("合成数据创建完成")
        return case_data, mobility_data, exposure_data
    
    def _create_synthetic_case_data(self):
        """创建模拟病例数据"""
        n_cases = 200
        start_date = datetime(2020, 1, 15)
        
        data = []
        for i in range(n_cases):
            case_id = f"case_{i:03d}"
            infection_date = start_date + timedelta(days=np.random.randint(0, 60))
            symptom_onset = infection_date + timedelta(days=np.random.randint(2, 10))
            confirmation_date = symptom_onset + timedelta(days=np.random.randint(1, 5))
            
            data.append({
                'case_id': case_id,
                'age': np.random.randint(20, 70),
                'gender': np.random.choice(['M', 'F']),
                'province': np.random.choice(['Guangdong', 'Zhejiang', 'Henan', 'Hunan']),
                'city': f"City_{np.random.randint(1, 10)}",
                'infection_date': infection_date.date(),
                'symptom_onset_date': symptom_onset.date(),
                'confirmation_date': confirmation_date.date(),
                'severity': np.random.choice(['mild', 'moderate', 'severe'], p=[0.7, 0.2, 0.1])
            })
        
        return pd.DataFrame(data)
    
    def _create_synthetic_mobility_data(self):
        """创建模拟移动性数据"""
        n_individuals = 500
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 3, 1)
        
        data = []
        current_id = 0
        
        for individual in range(n_individuals):
            individual_id = f"person_{individual:03d}"
            n_movements = np.random.randint(5, 20)
            
            current_date = start_date
            for move in range(n_movements):
                if current_date > end_date:
                    break
                    
                stay_duration = timedelta(days=np.random.randint(1, 7))
                location_type = np.random.choice(['home', 'work', 'shopping', 'transportation'])
                
                data.append({
                    'movement_id': current_id,
                    'individual_id': individual_id,
                    'date': current_date.date(),
                    'location_type': location_type,
                    'location_id': f"loc_{np.random.randint(1, 50)}",
                    'duration_hours': stay_duration.total_seconds() / 3600,
                    'city': f"City_{np.random.randint(1, 10)}"
                })
                
                current_id += 1
                current_date += stay_duration
        
        return pd.DataFrame(data)
    
    def _create_synthetic_exposure_data(self):
        """创建模拟暴露数据"""
        n_exposures = 1000
        start_date = datetime(2020, 1, 1)
        
        data = []
        for i in range(n_exposures):
            exposure_date = start_date + timedelta(days=np.random.randint(0, 60))
            duration = timedelta(hours=np.random.randint(1, 8))
            
            data.append({
                'exposure_id': i,
                'case_id': f"case_{np.random.randint(0, 200):03d}",
                'location_id': f"loc_{np.random.randint(1, 50)}",
                'exposure_start': exposure_date,
                'exposure_end': exposure_date + duration,
                'exposure_type': np.random.choice(['close_contact', 'same_location', 'transportation']),
                'risk_level': np.random.choice(['low', 'medium', 'high'])
            })
        
        return pd.DataFrame(data)
    
    def build_temporal_networks_from_real_data(self, case_data, mobility_data, exposure_data):
        """从真实数据构建时序网络"""
        print("从真实数据构建时序网络...")
        
        combined_data = self._combine_datasets(case_data, mobility_data, exposure_data)
        
        temporal_networks = {}
        date_range = pd.date_range(
            combined_data['date'].min(),
            combined_data['date'].max(),
            freq='D'
        )
        
        for date in tqdm(date_range, desc="构建时序网络"):
            daily_data = combined_data[combined_data['date'] == date.date()]
            
            if len(daily_data) > 0:
                network = self._build_daily_network(daily_data, date)
                temporal_networks[date] = network
        
        print(f"构建了 {len(temporal_networks)} 个时序网络")
        return temporal_networks
    
    def _combine_datasets(self, case_data, mobility_data, exposure_data):
        """合并多个数据集"""
        combined = []
        
        for _, row in mobility_data.iterrows():
            combined.append({
                'individual_id': row['individual_id'],
                'date': row['date'],
                'location': row['location_id'],
                'data_type': 'mobility',
                'risk_factor': 0.1
            })
        
        for _, row in exposure_data.iterrows():
            combined.append({
                'individual_id': row['case_id'],
                'date': row['exposure_start'].date(),
                'location': row['location_id'],
                'data_type': 'exposure',
                'risk_factor': 0.8 if row['risk_level'] == 'high' else 0.3
            })
        
        return pd.DataFrame(combined)
    
    def _build_daily_network(self, daily_data, date):
        """构建单日网络"""
        G = nx.Graph()
        G.graph['date'] = date
        G.graph['node_count'] = len(daily_data)
        
        for _, row in daily_data.iterrows():
            node_id = row['individual_id']
            G.add_node(node_id, **{
                'data_type': row['data_type'],
                'risk_factor': row.get('risk_factor', 0.0),
                'location': row.get('location', 'unknown')
            })
        
        location_groups = daily_data.groupby('location')
        edge_count = 0
        
        for location, group in location_groups:
            individuals = group['individual_id'].tolist()
            
            if len(individuals) >= 2:
                for i in range(len(individuals)):
                    for j in range(i + 1, len(individuals)):
                        node1 = individuals[i]
                        node2 = individuals[j]
                        
                        weight = self._calculate_edge_weight(
                            G.nodes[node1], G.nodes[node2]
                        )
                        
                        if weight > 0.1:
                            G.add_edge(node1, node2, 
                                    weight=weight,
                                    location=location,
                                    edge_type='co_location')
                            edge_count += 1
        
        if edge_count == 0 and len(daily_data) > 1:
            print(f"警告: {date.date()} 的网络没有边，添加弱连接")
            individuals = daily_data['individual_id'].tolist()
            for i in range(min(5, len(individuals))):
                for j in range(i + 1, min(i + 3, len(individuals))):
                    G.add_edge(individuals[i], individuals[j], 
                            weight=0.1, edge_type='weak_connection')
        
        print(f"  网络: {len(G.nodes())}节点, {G.number_of_edges()}边")
        return G
    
    def _calculate_edge_weight(self, node1_attrs, node2_attrs):
        """计算边权重"""
        weight = 0.0
        
        if node1_attrs.get('location') == node2_attrs.get('location'):
            weight += 0.5
        
        risk1 = node1_attrs.get('risk_factor', 0)
        risk2 = node2_attrs.get('risk_factor', 0)
        weight += (risk1 + risk2) * 0.3
        
        return min(weight, 1.0)