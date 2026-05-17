import requests
import os

def download_bird_plate_boundaries(save_filename="PB2002_boundaries.json"):
    """
    下载 Peter Bird (2003) 全球板块边界 GeoJSON 数据
    数据源: GitHub - fraxen/tectonicplates
    """
    # 指向 GitHub 上 GeoJSON 原生文件的 Raw 链接
    url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
    
    print(f"正在从 GitHub 下载 Peter Bird 板块边界数据...")
    print(f"数据源链接: {url}")
    
    try:
        # 发起 GET 请求，设置 30 秒超时
        response = requests.get(url, timeout=30)
        
        # 检查 HTTP 响应状态码，如果不是 200 则抛出异常
        response.raise_for_status() 
        
        # 将获取的二进制内容写入本地文件
        with open(save_filename, 'wb') as file:
            file.write(response.content)
            
        # 获取完整保存路径并打印成功信息
        full_path = os.path.abspath(save_filename)
        print(f"✅ 下载成功！")
        print(f"文件大小: {len(response.content) / 1024:.2f} KB")
        print(f"保存路径: {full_path}")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP 请求错误: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"❌ 网络连接错误 (可能需要检查网络或配置代理): {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"❌ 请求超时: {timeout_err}")
    except Exception as err:
        print(f"❌ 发生未知错误: {err}")

# ================= 脚本执行入口 =================
if __name__ == "__main__":
    download_bird_plate_boundaries()