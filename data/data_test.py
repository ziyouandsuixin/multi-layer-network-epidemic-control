import pandas as pd
import os
import numpy as np

def explore_real_data():
    """探索真实Nature数据集的结构"""
    data_path = "data/raw/dataset_EN.csv"
    
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return None
    
    # 加载数据，处理混合类型警告
    print("正在加载真实Nature数据集...")
    df = pd.read_csv(data_path, low_memory=False)
    
    print("\n" + "="*50)
    print("数据集基本信息")
    print("="*50)
    print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
    
    print(f"\n所有列名:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    print(f"\n前3行数据预览:")
    print(df.head(3))
    
    print(f"\n数据类型:")
    print(df.dtypes)
    
    print(f"\n尝试识别日期列:")
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    print(f"可能的日期列: {date_columns}")
    
    for col in date_columns:
        unique_dates = df[col].dropna().unique()
        print(f"\n{col} 的示例值:")
        print(f"  非空值数量: {len(unique_dates)}")
        if len(unique_dates) > 0:
            print(f"  前5个值: {unique_dates[:5]}")
    
    print(f"\n数值列统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"数值列: {list(numeric_cols)}")
        print(df[numeric_cols].describe())
    else:
        print("  没有数值列")
    
    print(f"\n关键分类列统计 (前10列):")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:10]:  # 只显示前10个分类列
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} 个唯一值")
        if unique_count <= 15:  # 如果唯一值较少，显示具体值
            print(f"    示例值: {df[col].unique()[:10]}")
    
    print(f"\n缺失值统计 (前20列):")
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0]
    if len(missing_cols) > 0:
        for col, count in list(missing_cols.items())[:20]:  # 只显示前20个有缺失值的列
            print(f"  {col}: {count} 个缺失值 ({count/len(df)*100:.1f}%)")
    else:
        print("  没有缺失值")
    
    # 特别检查可能的关键列
    print(f"\n检查可能的关键标识列:")
    potential_id_cols = [col for col in df.columns if 'id' in col.lower() or 'case' in col.lower()]
    for col in potential_id_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} 个唯一值")
    
    return df

# 执行探索
real_data = explore_real_data()