from __future__ import annotations

import argparse
import calendar
import gzip
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GCMT_CATALOG_BASE_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog"
GCMT_LEGACY_ARCHIVE_URL = f"{GCMT_CATALOG_BASE_URL}/jan76_dec20.ndk.gz"
GCMT_MONTHLY_BASE_URL = f"{GCMT_CATALOG_BASE_URL}/NEW_MONTHLY"

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


def _parse_ndk_header_time(header: list[str]) -> tuple[pd.Timestamp, float, float, float]:
    """解析 NDK 第一行中的时间、纬度、经度和深度。"""
    if len(header) < 7:
        return pd.NaT, np.nan, np.nan, np.nan

    try:
        if "/" in header[1]:
            # 标准 Global CMT NDK: PDEW 2024/01/01 07:10:09.5 ...
            date_token = header[1]
            time_token = header[2]
            lat = float(header[3])
            lon = float(header[4])
            depth = float(header[5])
        else:
            # 兼容旧解析逻辑: PDEW 2024 1 1 07 10 09.5 ...
            date_token = f"{int(header[1]):04d}/{int(header[2]):02d}/{int(header[3]):02d}"
            time_token = f"{int(header[4]):02d}:{int(header[5]):02d}:{float(header[6]):04.1f}"
            lat = float(header[7])
            lon = float(header[8])
            depth = float(header[9])
    except (ValueError, IndexError):
        return pd.NaT, np.nan, np.nan, np.nan

    event_time = pd.to_datetime(
        f"{date_token} {time_token}",
        utc=True,
        errors="coerce",
        format="mixed",
    )
    return event_time, lat, lon, depth


def _parse_tensor_line(line: str) -> tuple[float, float, float, float, float, float, int]:
    """解析 NDK 第 4 行中的矩张量分量。"""
    parts = line.split()
    if len(parts) < 12:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    try:
        exponent = int(float(parts[0]))
    except ValueError:
        exponent = 0

    scale = 10.0 ** exponent
    values = []
    for idx in (1, 3, 5, 7, 9, 11):
        values.append(_parse_float(parts[idx]) * scale if idx < len(parts) else np.nan)
    return (*values, exponent)


def _parse_principal_axis_line(line: str, exponent: int) -> dict:
    """解析 NDK 第 5 行中的 P/T/B 轴、标量矩和节点面。"""
    parts = line.split()
    result = {
        "mw": np.nan,
        "scalar_moment": np.nan,
        "plunge_T": np.nan,
        "trend_T": np.nan,
        "plunge_B": np.nan,
        "trend_B": np.nan,
        "plunge_P": np.nan,
        "trend_P": np.nan,
        "strike1": np.nan,
        "dip1": np.nan,
        "rake1": np.nan,
        "strike2": np.nan,
        "dip2": np.nan,
        "rake2": np.nan,
        "f_clvd": np.nan,
    }
    if len(parts) < 17:
        return result

    eig_t = _parse_float(parts[1])
    eig_b = _parse_float(parts[4])
    eig_p = _parse_float(parts[7])
    m0_mantissa = _parse_float(parts[10])
    m0_dyne_cm = m0_mantissa * (10.0 ** exponent) if np.isfinite(m0_mantissa) else np.nan

    if np.isfinite(m0_dyne_cm) and m0_dyne_cm > 0:
        result["scalar_moment"] = float(m0_dyne_cm)
        # Global CMT 的标量矩单位为 dyne-cm: Mw = 2/3 * (log10(M0) - 16.1)
        result["mw"] = float((2.0 / 3.0) * (np.log10(m0_dyne_cm) - 16.1))

    result.update(
        {
            "plunge_T": _parse_float(parts[2]),
            "trend_T": _parse_float(parts[3]),
            "plunge_B": _parse_float(parts[5]),
            "trend_B": _parse_float(parts[6]),
            "plunge_P": _parse_float(parts[8]),
            "trend_P": _parse_float(parts[9]),
            "strike1": _parse_float(parts[11]),
            "dip1": _parse_float(parts[12]),
            "rake1": _parse_float(parts[13]),
            "strike2": _parse_float(parts[14]),
            "dip2": _parse_float(parts[15]),
            "rake2": _parse_float(parts[16]),
        }
    )

    if np.isfinite(eig_t) and abs(eig_t) > 1e-30 and np.isfinite(eig_b):
        result["f_clvd"] = float(-eig_b / max(abs(eig_t), 1e-30))

    return result


def parse_gcmt_ndk(text: str) -> pd.DataFrame:
    """解析 Global CMT NDK 格式文本为 DataFrame。"""
    records: list[dict] = []
    lines = text.strip().split("\n")
    data_lines = [ln.rstrip("\n") for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    i = 0
    while i <= len(data_lines) - 5:
        header = data_lines[i].split()
        if len(header) < 6 or not re.search(r"\d{4}/\d{2}/\d{2}|\d{4}/\d{1,2}/\d{1,2}", data_lines[i]):
            i += 1
            continue

        event_time, lat, lon, depth = _parse_ndk_header_time(header)
        if pd.isna(event_time) or not np.isfinite([lat, lon, depth]).all():
            i += 1
            continue

        event_line = data_lines[i + 1].strip()
        event_name_parts = event_line.split()
        event_name = event_name_parts[0] if event_name_parts else ""
        mrr, mtt, mpp, mrt, mrp, mtp, exponent = _parse_tensor_line(data_lines[i + 3])
        axis_features = _parse_principal_axis_line(data_lines[i + 4], exponent)

        records.append({
            "time": event_time,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "mw": axis_features["mw"],
            "scalar_moment": axis_features["scalar_moment"],
            "mrr": mrr,
            "mtt": mtt,
            "mpp": mpp,
            "mrt": mrt,
            "mrp": mrp,
            "mtp": mtp,
            "event_name": event_name,
            "strike1": axis_features["strike1"],
            "dip1": axis_features["dip1"],
            "rake1": axis_features["rake1"],
            "strike2": axis_features["strike2"],
            "dip2": axis_features["dip2"],
            "rake2": axis_features["rake2"],
            "plunge_P": axis_features["plunge_P"],
            "trend_P": axis_features["trend_P"],
            "plunge_T": axis_features["plunge_T"],
            "trend_T": axis_features["trend_T"],
            "plunge_B": axis_features["plunge_B"],
            "trend_B": axis_features["trend_B"],
            "f_clvd": axis_features["f_clvd"],
        })

        i += 5

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
    parsed_cols = {
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
        "plunge_P", "trend_P", "plunge_T", "trend_T", "plunge_B", "trend_B",
        "f_clvd",
    }
    if parsed_cols.issubset(df.columns) and df["strike1"].notna().any():
        out = df.copy()
        out["fault_type"] = out["rake1"].map(_classify_fault_type).fillna("UNK")
        keep_cols = [
            "time", "latitude", "longitude", "depth", "mw", "event_name",
            "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
            "fault_type", "plunge_P", "trend_P", "plunge_T", "trend_T",
            "plunge_B", "trend_B", "f_clvd",
        ]
        return out[keep_cols]

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
            "strike2": strike2, "dip2": dip2, "rake2": rake2,
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

    def _response_to_text(resp: requests.Response, url: str) -> str:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=BytesIO(resp.content)) as file:
                return file.read().decode("utf-8", errors="replace")
        return resp.text

    def _download_text(urls: list[str]) -> tuple[str, str] | None:
        for url in urls:
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                return _response_to_text(resp, url), url
            except (requests.RequestException, OSError, gzip.BadGzipFile):
                continue
        return None

    if start_year <= 2020:
        end_filter = min(end_year, 2020)
        print(f"  [1976-2020] 下载历史归档 ...", end=" ", flush=True)
        downloaded = _download_text([GCMT_LEGACY_ARCHIVE_URL])
        if downloaded is None:
            print("失败")
        else:
            text, url = downloaded
            df_archive = parse_gcmt_ndk(text)
            if not df_archive.empty:
                times = pd.to_datetime(df_archive["time"], utc=True, errors="coerce")
                df_archive = df_archive.loc[
                    (times.dt.year >= start_year) & (times.dt.year <= end_filter)
                ].copy()
                if not df_archive.empty:
                    all_dfs.append(df_archive)
            print(f"✓ {len(df_archive) if 'df_archive' in locals() else 0} 条 ({url})")

    monthly_start_year = max(start_year, 2021)
    for year in range(monthly_start_year, end_year + 1):
        for month in range(1, 13):
            month_name = calendar.month_abbr[month].lower()
            yy = str(year)[-2:]
            urls = [
                f"{GCMT_MONTHLY_BASE_URL}/{year}/{month_name}{yy}.ndk",
                f"{GCMT_MONTHLY_BASE_URL}/{year}/{month_name}{yy}.ndk.gz",
            ]
            print(f"  [{year}-{month:02d}] 尝试下载 ...", end=" ", flush=True)
            downloaded = _download_text(urls)
            if downloaded is None:
                print("无数据")
                time.sleep(request_sleep)
                continue
            text, source_url = downloaded

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
