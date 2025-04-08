import os
import pandas as pd
from config.datasetconfig import mapping
def merge_csv_files(root_dir):
    # 存储所有csv文件的路径
    csv_files = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 读取并合并所有CSV文件
    all_df = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # 去掉空格
            df.columns = df.columns.str.strip()
            
            
            # 应用映射
            df.rename(columns=mapping, inplace=True)
            # df.drop(columns=['Fwd_Header_Length.1'], inplace=True)
            # 再次确保没有空格
            # df.columns = df.columns.str.strip()
            # df.columns = df.columns.str.replace(' ', '_')
            
            all_df.append(df)
            print(f"成功读取: {file}")
        except Exception as e:
            print(f"读取文件出错 {file}: {str(e)}")
    
    # 合并所有数据框
    if all_df:
        merged_df = pd.concat(all_df, ignore_index=True)
        merged_df.drop(columns=['Fwd Header Length.1'], inplace=True)
        # 保存合并后的文件
        output_file = 'mergedata/merged_data.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\n合并完成! 保存到: {output_file}")
        print(f"合并后的数据大小: {merged_df.shape}")
        print(merged_df.columns)
    else:
        print("没有找到可以合并的CSV文件")

if __name__ == "__main__":
    data_dir = "datae"
    merge_csv_files(data_dir)