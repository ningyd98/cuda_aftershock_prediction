from __future__ import annotations

import argparse
import gzip
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GCMT_BASE_URL = "https://www.globalcmt.org/CMTfiles/NCEQ"

# ———— NDK 格式解析 ————
# Global CMT 使用 NDK (NCEQ Deci-Kilometer) 格式，每个事件占 4-5 行。
# 参照 https://www.globalcmt.org/CMTfiles/NCEQ/README

_GCMT_EVENT_RE = re.compile(
    r"PDEW?\s+\d{4}\s+\d{1,2}\s+\d{1,2}\s+\d{2}\s+\d{2}\s+\d{2}\.\d+"
)


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def parse_gcmt_ndk(text: str) -> pd.DataFrame:
    """解析 Global CMT NDK 格式文本为 DataFrame。"""
    records: list[dict] = []
    # NDK 每 5 行为一个事件（第5行可能为空或为注释）
    lines = text.strip().split("\n")
    # 过滤注释行
    data_lines = [ln for ln in lines if not ln.strip().startswith("#")]

    i = 0
    while i < len(data_lines) - 3:
        # 第1行: 头信息 (PDE 标识, 时间, 位置, 震级等)
        header = data_lines[i].split()
        if len(header) < 10:
            i += 1
            continue

        try:
            year = int(header[1])
            month = int(header[2])
            day = int(header[3])
            hour = int(header[4])
            minute = int(header[5])
            second = float(header[6])
            lat = float(header[7])
            lon = float(header[8])
            depth = float(header[9])
        except (ValueError, IndexError):
            i += 1
            continue

        # 第2行: 事件名 + 矩张量分量 (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
        if i + 3 >= len(data_lines):
            break
        event_line = data_lines[i + 1].strip()
        # 事件名通常在行首
        event_name_parts = event_line.split()
        event_name = event_name_parts[0] if event_name_parts else ""
        # 矩张量分量在行尾（最后12列通常是6个指数+6个尾数）
        # NDK 格式: BXXXXXXA BXXXXXXA ... 每个12字符含一位指数

        # 第3行: 更多事件信息 + 指数
        # 第4行: 矩震级 + 标量矩 + 半持续时间 + 时间延迟 + 震源时间函数类型

        # 实际上 NDK 格式非常复杂，这里采用简化解析：
        # 用正则从2-4行提取 Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
        tensor_text = " ".join(data_lines[i + 1 : i + 4])
        # 匹配形如 -1.230e+19 的科学记数法
        tensor_values = re.findall(r"(-?\d+\.\d+e[+-]\d+)", tensor_text)
        mrr = mtt = mpp = mrt = mrp = mtp = np.nan
        if len(tensor_values) >= 6:
            mrr = float(tensor_values[0])
            mtt = float(tensor_values[1])
            mpp = float(tensor_values[2])
            mrt = float(tensor_values[3])
            mrp = float(tensor_values[4])
            mtp = float(tensor_values[5])

        # 第4行: 版本码 + Mw + 标量矩
        line4 = data_lines[i + 3].split()
        mw = _parse_float(line4[1]) if len(line4) > 1 else np.nan
        scalar_moment = _parse_float(line4[2]) if len(line4) > 2 else np.nan

        # 构建时间
        try:
            time_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{int(second):02d}"
            event_time = pd.to_datetime(time_str, utc=True)
        except Exception:
            event_time = pd.NaT

        records.append({
            "time": event_time,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "mw": mw,
            "scalar_moment": scalar_moment,
            "mrr": mrr,
            "mtt": mtt,
            "mpp": mpp,
            "mrt": mrt,
            "mrp": mrp,
            "mtp": mtp,
            "event_name": event_name,
        })

        i += 5  # 跳过已解析的 5 行

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.dropna(subset=["time", "latitude", "longitude"]).sort_values("time").reset_index(drop=True)
    return df


def compute_focal_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从矩张量分量计算震源机制解特征。

    返回: strike1, dip1, rake1, strike2, dip2, rake2,
          fault_type (Normal/Strike-slip/Reverse/Oblique),
          plunge_P, trend_P, plunge_T, trend_T, plunge_B, trend_B,
          f_clvd (CLVD 分量比例)
    """
    from numpy.linalg import eigh

    results: list[dict] = []
    for _, row in df.iterrows():
        mrr = row.get("mrr", np.nan)
        mtt = row.get("mtt", np.nan)
        mpp = row.get("mpp", np.nan)
        mrt = row.get("mrt", np.nan)
        mrp = row.get("mrp", np.nan)
        mtp = row.get("mtp", np.nan)

        if not all(np.isfinite([mrr, mtt, mpp, mrt, mrp, mtp])):
            results.append({
                "strike1": np.nan, "dip1": np.nan, "rake1": np.nan,
                "strike2": np.nan, "dip2": np.nan, "rake2": np.nan,
                "fault_type": "UNK",
                "plunge_P": np.nan, "trend_P": np.nan,
                "plunge_T": np.nan, "trend_T": np.nan,
                "plunge_B": np.nan, "trend_B": np.nan,
                "f_clvd": np.nan,
            })
            continue

        # 构建矩张量矩阵 (NEZ 坐标系 → NED: r=up, t=south, p=east → x=north, y=east, z=down)
        # Global CMT 使用 up, south, east (USE) 坐标系
        # M_USE → M_NED: Mrr_NED = Mrr, Mtt_NED = Mpp, Mpp_NED = Mtt,
        #                   Mrt_NED = -Mrp, Mrp_NED = -Mrt, Mtp_NED = -Mtp
        M_ned = np.array([
            [mrr, -mrp, -mrt],
            [-mrp, mpp, -mtp],
            [-mrt, -mtp, mtt],
        ], dtype=float)

        # 特征值分解
        eigvals, eigvecs = eigh(M_ned)
        # 排序：|λ1| ≥ |λ2| ≥ |λ3|
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 检查 trace ~ 0 (双力偶条件)
        trace = np.trace(M_ned)
        is_dc = abs(trace) < 1e-3 * max(abs(eigvals[0]), 1.0)

        # P/T/B 轴
        T_vec = eigvecs[:, 0]  # 最大正特征值
        P_vec = eigvecs[:, 2]  # 最小负特征值 (最压缩)
        B_vec = eigvecs[:, 1]  # 中间

        def vec_to_plunge_trend(v: np.ndarray) -> tuple[float, float]:
            """特征向量 → (plunge_deg, trend_deg)"""
            v = v / np.linalg.norm(v)
            # v = [N, E, D] (North, East, Down)
            vn, ve, vd = v[0], v[1], v[2]
            plunge = np.degrees(np.arcsin(max(-1.0, min(1.0, -vd))))
            trend = np.degrees(np.arctan2(ve, vn))
            trend = (trend + 360.0) % 360.0
            return float(plunge), float(trend)

        plunge_T, trend_T = vec_to_plunge_trend(T_vec)
        plunge_P, trend_P = vec_to_plunge_trend(P_vec)
        plunge_B, trend_B = vec_to_plunge_trend(B_vec)

        # 节点面 (从矩张量出发的经典分解)
        # 简化：用 P/T 轴反推节点面
        # 方法参考 Aki & Richards (1980) / Jost & Herrmann (1989)
        try:
            strike1, dip1, rake1, strike2, dip2, rake2 = _mt_to_fault_planes(
                mrr, mtt, mpp, mrt, mrp, mtp
            )
        except Exception:
            strike1 = dip1 = rake1 = strike2 = dip2 = rake2 = np.nan

        # 断层类型分类
        fault_type = _classify_fault_type(rake1) if np.isfinite(rake1) else "UNK"

        # CLVD 分量比例
        eps = max(abs(eigvals[0]), 1e-30)
        f_clvd = float(-eigvals[1] / eps) if is_dc else np.nan

        results.append({
            "strike1": strike1, "dip1": dip1, "rake1": rake1,
            "strike2": strike2, "dip2": dip2, "rake2": dip2,
            "fault_type": fault_type,
            "plunge_P": plunge_P, "trend_P": trend_P,
            "plunge_T": plunge_T, "trend_T": trend_T,
            "plunge_B": plunge_B, "trend_B": trend_B,
            "f_clvd": f_clvd,
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df[["time", "latitude", "longitude", "depth", "mw", "event_name"]], result_df], axis=1)


def _mt_to_fault_planes(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    从矩张量分量计算双力偶节点面参数 (走向/倾角/滑动角)。

    采用 Jost & Herrmann (1989) 方法。坐标系: USE (Up/South/East)
    """
    # 标准化
    M0 = np.sqrt(mrr**2 + mtt**2 + mpp**2 + 2*(mrt**2 + mrp**2 + mtp**2))
    if M0 < 1e-30:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # USE → NED 转换 (已在调用方完成，这里假设输入已是 NED)
    # 构建矩张量
    M = np.array([[mrr, mrt, mrp],
                   [mrt, mtt, mtp],
                   [mrp, mtp, mpp]], dtype=float)

    # 特征分解
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(np.abs(vals))[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # T (最大) 和 P (最小) 轴
    T = vecs[:, 0]
    P = vecs[:, 2]

    # 节点面法向量 = (T + P)/√2, 滑动向量 = (T - P)/√2
    n1 = (T + P) / np.sqrt(2)
    d1 = (T - P) / np.sqrt(2)
    n2 = (T - P) / np.sqrt(2)
    d2 = (T + P) / np.sqrt(2)

    def normal_to_strike_dip(n):
        """法向量 → (strike, dip)"""
        n = n / np.linalg.norm(n)
        # n = [N, E, D] (NED)
        n_n, n_e, n_d = n[0], n[1], n[2]
        dip = np.degrees(np.arccos(max(-1.0, min(1.0, -n_d))))
        strike_rad = np.arctan2(-n_n, n_e)
        strike = (np.degrees(strike_rad) + 360.0) % 360.0
        return float(strike), float(dip)

    def rake_from_sd(s, d):
        """从滑动向量和法向量计算 rake。"""
        n = s / np.linalg.norm(s)
        d_unit = d / np.linalg.norm(d)
        # 确保滑动向量垂直于法向量
        d_proj = d_unit - np.dot(d_unit, n) * n
        norm_dp = np.linalg.norm(d_proj)
        if norm_dp < 1e-10:
            return float(np.nan)
        d_proj = d_proj / norm_dp
        # 计算 rake
        sin_rake = -np.dot(np.cross(n, d_proj), n)
        cos_rake = np.dot(d_proj, d_unit)
        rake = np.degrees(np.arctan2(sin_rake, cos_rake))
        return float(rake)

    strike1, dip1 = normal_to_strike_dip(n1)
    strike2, dip2 = normal_to_strike_dip(n2)

    rake1 = rake_from_sd(n1, d1)
    rake2 = rake_from_sd(n2, d2)

    return strike1, dip1, rake1, strike2, dip2, rake2


def _classify_fault_type(rake: float) -> str:
    """根据滑动角分类断层类型。"""
    if not np.isfinite(rake):
        return "UNK"
    abs_rake = abs(rake)
    if abs_rake <= 30 or abs_rake >= 150:
        return "SS"  # Strike-slip
    if rake > 0:
        if abs_rake >= 120:
            return "SS"
        if abs_rake >= 60:
            return "TF"  # Thrust/Reverse
        return "NF"  # Normal
    else:
        if abs_rake >= 120:
            return "SS"
        if abs_rake >= 60:
            return "NF"
        return "TF"


def download_gcmt_catalog(
    start_year: int = 1976,
    end_year: int = 2024,
    output_dir: Path | None = None,
    request_sleep: float = 0.5,
) -> pd.DataFrame:
    """下载 Global CMT 目录 (1976-) 并解析保存。"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        # 月文件
        for month in range(1, 13):
            url = f"{GCMT_BASE_URL}/{year}{month:02d}.ndk"
            print(f"  [{year}-{month:02d}] 尝试下载 ...", end=" ", flush=True)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    # 尝试 gz 压缩格式
                    url_gz = f"{url}.gz"
                    resp = requests.get(url_gz, timeout=30)
                    if resp.status_code == 404:
                        print("无数据")
                        time.sleep(request_sleep)
                        continue
                    resp.raise_for_status()
                    # 解压
                    with gzip.GzipFile(fileobj=BytesIO(resp.content)) as f:
                        text = f.read().decode("utf-8", errors="replace")
                else:
                    resp.raise_for_status()
                    text = resp.text
            except requests.RequestException as exc:
                print(f"失败: {exc}")
                time.sleep(request_sleep)
                continue

            try:
                df_month = parse_gcmt_ndk(text)
                if not df_month.empty:
                    all_dfs.append(df_month)
                    print(f"✓ {len(df_month)} 条")
                else:
                    print("解析为空")
            except Exception as exc:
                print(f"解析异常: {exc}")

            time.sleep(request_sleep)

    if not all_dfs:
        raise RuntimeError("未获取到任何 Global CMT 数据。")

    catalog_df = pd.concat(all_dfs, ignore_index=True)
    catalog_df = catalog_df.sort_values("time").drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)

    # 计算震源机制特征
    catalog_df = compute_focal_mechanism_features(catalog_df)

    out_path = output_dir / f"GlobalCMT_{start_year}-{end_year}.csv"
    catalog_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✓ Global CMT 目录已保存: {out_path}")
    print(f"  事件总数: {len(catalog_df)}")
    print(f"  时间跨度: {catalog_df['time'].min()} ~ {catalog_df['time'].max()}")
    return catalog_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载解析 Global CMT 震源机制解目录")
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_gcmt_catalog(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
