mapping = {
    'Flow ID': 'Flow_ID',                          # 流量ID
    'Src IP': 'Source_IP',                         # 源IP地址
    'Src Port': 'Source_Port',                     # 源端口
    'Dst IP': 'Destination_IP',                    # 目标IP地址
    'Dst Port': 'Destination_Port',                # 目标端口
    'Protocol': 'Protocol',                        # 协议
    'Timestamp': 'Timestamp',                      # 时间戳
    'Flow Duration': 'Flow_Duration',              # 流量持续时间
    'Total Fwd Packet': 'Total_Fwd_Packets',      # 总前向数据包数
    'Total Bwd packets': 'Total_Backward_Packets',  # 总后向数据包数
    'Total Length of Fwd Packet': 'Total_Length_of_Fwd_Packets',  # 前向数据包总长度
    'Total Length of Bwd Packet': 'Total_Length_of_Bwd_Packets',  # 后向数据包总长度
    'Fwd Packet Length Max': 'Fwd_Packet_Length_Max',  # 前向数据包最大长度
    'Fwd Packet Length Min': 'Fwd_Packet_Length_Min',  # 前向数据包最小长度
    'Fwd Packet Length Mean': 'Fwd_Packet_Length_Mean',  # 前向数据包平均长度
    'Fwd Packet Length Std': 'Fwd_Packet_Length_Std',    # 前向数据包长度标准差
    'Bwd Packet Length Max': 'Bwd_Packet_Length_Max',    # 后向数据包最大长度
    'Bwd Packet Length Min': 'Bwd_Packet_Length_Min',    # 后向数据包最小长度
    'Bwd Packet Length Mean': 'Bwd_Packet_Length_Mean',  # 后向数据包平均长度
    'Bwd Packet Length Std': 'Bwd_Packet_Length_Std',    # 后向数据包长度标准差
    'Flow Bytes/s': 'Flow_Bytes/s',                      # 每秒流量字节数
    'Flow Packets/s': 'Flow_Packets/s',                  # 每秒流量数据包数
    'Flow IAT Mean': 'Flow_IAT_Mean',                    # 流量数据包到达间隔平均值
    'Flow IAT Std': 'Flow_IAT_Std',                      # 流量数据包到达间隔标准差
    'Flow IAT Max': 'Flow_IAT_Max',                      # 流量数据包到达间隔最大值
    'Flow IAT Min': 'Flow_IAT_Min',                      # 流量数据包到达间隔最小值
    'Fwd IAT Total': 'Fwd_IAT_Total',                    # 前向数据包到达间隔总和
    'Fwd IAT Mean': 'Fwd_IAT_Mean',                      # 前向数据包到达间隔平均值
    'Fwd IAT Std': 'Fwd_IAT_Std',                        # 前向数据包到达间隔标准差
    'Fwd IAT Max': 'Fwd_IAT_Max',                        # 前向数据包到达间隔最大值
    'Fwd IAT Min': 'Fwd_IAT_Min',                        # 前向数据包到达间隔最小值
    'Bwd IAT Total': 'Bwd_IAT_Total',                    # 后向数据包到达间隔总和
    'Bwd IAT Mean': 'Bwd_IAT_Mean',                      # 后向数据包到达间隔平均值
    'Bwd IAT Std': 'Bwd_IAT_Std',                        # 后向数据包到达间隔标准差
    'Bwd IAT Max': 'Bwd_IAT_Max',                        # 后向数据包到达间隔最大值
    'Bwd IAT Min': 'Bwd_IAT_Min',                        # 后向数据包到达间隔最小值
    'Fwd PSH Flags': 'Fwd_PSH_Flags',                    # 前向PSH标志
    'Bwd PSH Flags': 'Bwd_PSH_Flags',                    # 后向PSH标志
    'Fwd URG Flags': 'Fwd_URG_Flags',                    # 前向URG标志
    'Bwd URG Flags': 'Bwd_URG_Flags',                    # 后向URG标志
    'Fwd Header Length': 'Fwd_Header_Length',            # 前向头部长度
    'Bwd Header Length': 'Bwd_Header_Length',            # 后向头部长度
    'Fwd Packets/s': 'Fwd_Packets/s',                    # 每秒前向数据包数
    'Bwd Packets/s': 'Bwd_Packets/s',                    # 每秒后向数据包数
    'Min Packet Length': 'Min_Packet_Length',            # 最小数据包长度
    'Max Packet Length': 'Max_Packet_Length',            # 最大数据包长度
    'Packet Length Mean': 'Packet_Length_Mean',          # 数据包平均长度
    'Packet Length Std': 'Packet_Length_Std',            # 数据包长度标准差
    'Packet Length Variance': 'Packet_Length_Variance',  # 数据包长度方差
    'FIN Flag Count': 'FIN_Flag_Count',                  # FIN标志计数
    'SYN Flag Count': 'SYN_Flag_Count',                  # SYN标志计数
    'RST Flag Count': 'RST_Flag_Count',                  # RST标志计数
    'PSH Flag Count': 'PSH_Flag_Count',                  # PSH标志计数
    'ACK Flag Count': 'ACK_Flag_Count',                  # ACK标志计数
    'URG Flag Count': 'URG_Flag_Count',                  # URG标志计数
               
    'ECE Flag Count': 'ECE_Flag_Count',                  # ECE标志计数
    'Down/Up Ratio': 'Down/Up_Ratio',                    # 下行/上行比率
    'Average Packet Size': 'Average_Packet_Size',        # 平均数据包大小
    'Avg Fwd Segment Size': 'Avg_Fwd_Segment_Size',      # 平均前向段大小
    'Avg Bwd Segment Size': 'Avg_Bwd_Segment_Size',      # 平均后向段大小
    # 前向头部长度（重复字段）
    'Fwd Avg Bytes/Bulk': 'Fwd_Avg_Bytes/Bulk',          # 前向平均每批量字节数
    'Fwd Avg Packets/Bulk': 'Fwd_Avg_Packets/Bulk',      # 前向平均每批量数据包数
    'Fwd Avg Bulk Rate': 'Fwd_Avg_Bulk_Rate',            # 前向平均批量速率
    'Bwd Avg Bytes/Bulk': 'Bwd_Avg_Bytes/Bulk',          # 后向平均每批量字节数
    'Bwd Avg Packets/Bulk': 'Bwd_Avg_Packets/Bulk',      # 后向平均每批量数据包数
    'Bwd Avg Bulk Rate': 'Bwd_Avg_Bulk_Rate',            # 后向平均批量速率
    'Subflow Fwd Packets': 'Subflow_Fwd_Packets',        # 子流前向数据包数
    'Subflow Fwd Bytes': 'Subflow_Fwd_Bytes',            # 子流前向字节数
    'Subflow Bwd Packets': 'Subflow_Bwd_Packets',        # 子流后向数据包数
    'Subflow Bwd Bytes': 'Subflow_Bwd_Bytes',            # 子流后向字节数
    'Init Win bytes forward': 'Init_Win_bytes_forward',  # 前向初始窗口字节数
    'Init Win bytes backward': 'Init_Win_bytes_backward',  # 后向初始窗口字节数
    'act data pkt fwd': 'act_data_pkt_fwd',              # 前向实际数据包数
    'min seg size forward': 'min_seg_size_forward',      # 前向最小段大小
    'Active Mean': 'Active_Mean',                        # 活跃时间平均值
    'Active Std': 'Active_Std',                          # 活跃时间标准差
    'Active Max': 'Active_Max',                          # 活跃时间最大值
    'Active Min': 'Active_Min',                          # 活跃时间最小值
    'Idle Mean': 'Idle_Mean',                            # 空闲时间平均值
    'Idle Std': 'Idle_Std',                              # 空闲时间标准差
    'Idle Max': 'Idle_Max',                              # 空闲时间最大值
    'Idle Min': 'Idle_Min',                              # 空闲时间最小值
    'Label': 'Label'       ,                              # 标签
    'Packet Length Min': 'Min_Packet_Length',
    'Packet Length Max': 'Max_Packet_Length',
    'CWR Flag Count': 'CWR_Flag_Count',
    'Fwd Segment Size Avg': 'Avg_Fwd_Segment_Size',
    'Bwd Segment Size Avg': 'Avg_Bwd_Segment_Size',
    'Fwd Bytes/Bulk Avg': 'Fwd_Avg_Bytes/Bulk',
    'Fwd Packet/Bulk Avg': 'Fwd_Avg_Packets/Bulk',
    'Fwd Bulk Rate Avg': 'Fwd_Avg_Bulk_Rate',
    'Bwd Bytes/Bulk Avg': 'Bwd_Avg_Bytes/Bulk',
    'Bwd Packet/Bulk Avg': 'Bwd_Avg_Packets/Bulk',
    'Bwd Bulk Rate Avg': 'Bwd_Avg_Bulk_Rate',
    'FWD Init Win Bytes': 'Init_Win_bytes_forward',
    'Bwd Init Win Bytes': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min': 'min_seg_size_forward',

    


}