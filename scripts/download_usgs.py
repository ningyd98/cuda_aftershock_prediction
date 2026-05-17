import pandas as pd
import requests
from io import StringIO
import time

def fetch_usgs_earthquakes(start_year, end_year, min_mag=6.0, max_depth=70.0):
    """
    通过 USGS API 获取指定条件的历史地震数据。
    
    参数:
    start_year (int): 起始年份
    end_year (int): 结束年份
    min_mag (float): 最小震级 (大赛要求为 6.0)
    max_depth (float): 最大震源深度 (单位: km, 大赛要求浅于 70km)
    """
    
    # USGS 地震查询官方 API 接口
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_data_frames = []

    print(f"开始获取 {start_year} 年至 {end_year} 年的全球强震数据...")
    
    # 按年份循环请求，避免触发单次请求数据量上限
    for year in range(start_year, end_year + 1):
        print(f"正在拉取 {year} 年的数据...", end=" ")
        
        # 构建 API 参数
        params = {
            "format": "csv",               # 直接请求 CSV 格式方便 pandas 处理
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": min_mag,
            "maxdepth": max_depth,
            "orderby": "time"              # 按时间先后排序
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            # 检查请求是否成功
            if response.status_code == 200:
                # 使用 StringIO 将返回的文本流转化为 pandas DataFrame
                df_year = pd.read_csv(StringIO(response.text))
                all_data_frames.append(df_year)
                print(f"成功！获取到 {len(df_year)} 条记录。")
            else:
                print(f"失败！状态码: {response.status_code}")
                
        except Exception as e:
            print(f"请求异常: {e}")

        # 礼貌性延迟，避免给 USGS 服务器造成过大压力被封 IP
        time.sleep(1)

    # 合并所有年份的数据
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        # 将时间列转换为 datetime 对象，方便后续时序分析
        final_df['time'] = pd.to_datetime(final_df['time'])
        return final_df
    else:
        print("未获取到任何数据。")
        return pd.DataFrame()

# ================= 脚本执行入口 =================

if __name__ == "__main__":
    # 设定时间范围：获取 1970 年至今的高质量仪器记录数据
    START_YEAR = 1970
    END_YEAR = 2023
    
    # 执行数据拉取
    df_earthquakes = fetch_usgs_earthquakes(
        start_year=START_YEAR, 
        end_year=END_YEAR, 
        min_mag=6.0, 
        max_depth=70.0
    )

    if not df_earthquakes.empty:
        print("\n--- 数据获取完成 ---")
        print(f"总计获取地震记录数: {len(df_earthquakes)}")
        
        # 提取对赛题最有用的核心字段进行预览
        core_columns = ['time', 'latitude', 'longitude', 'depth', 'mag', 'magType', 'place']
        print("\n数据预览 (前 5 行):")
        print(df_earthquakes[core_columns].head())

        # 保存为 CSV 文件至本地
        output_filename = f"USGS_Mw6.0_Depth70_{START_YEAR}-{END_YEAR}.csv"
        df_earthquakes.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\n完整数据已成功保存至当前目录: {raw_data/output_filename}")