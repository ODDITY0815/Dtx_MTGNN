# convert_and_check.py
from __future__ import annotations
import argparse, os, re, glob
import numpy as np
import pandas as pd

VAR_COLUMNS_DEFAULT = ["dstot", "eu_lag_sum", "hr_mean", "hr_var", "pain_mean"]

# --------------------------
# 유틸
# --------------------------
def _apply_hour_policy(series: pd.Series, policy: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if policy == "floor":
        return np.floor(s).astype(int)
    elif policy == "round":
        return np.rint(s).astype(int)
    elif policy == "ceil":
        return np.ceil(s).astype(int)
    else:
        raise ValueError(f"Unknown hour_policy: {policy}")

def read_csv_to_frame(
    csv_path: str,
    day_col: str = "day",
    hour_col: str = "hour_t",
    var_columns: list[str] = VAR_COLUMNS_DEFAULT,
    interpolate: bool = True,
    hour_policy: str = "round",
) -> pd.DataFrame:
    """
    CSV -> (T,N) DataFrame (DatetimeIndex)
    'day'는 상대 일수(1,2,3,...)로 가정하고 기준일 + 일 오프셋, hour_t는 시간 오프셋.
    """
    df = pd.read_csv(csv_path)

    # (1) 기준일 + 오프셋 (day, hour_t)
    base_date  = pd.Timestamp("2000-01-01")
    day_offset = pd.to_timedelta(pd.to_numeric(df[day_col], errors="coerce").fillna(0).astype(int), unit="D")
    hour       = _apply_hour_policy(df[hour_col], policy=hour_policy)
    hour_off   = pd.to_timedelta(hour, unit="h")
    time_index = base_date + day_offset + hour_off

    # (2) 변수 확인
    missing = [c for c in var_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}에 없는 컬럼: {missing}")

    # (3) 값/인덱스/정렬
    values = df[var_columns].copy()
    values.index = time_index
    values = values.sort_index()

    # (4) 결측 보간(옵션)
    if interpolate:
        values = values.interpolate(method="time", limit_direction="both")

    return values  # (T,N)

def generate_graph_seq2seq_io_data(
    df: pd.DataFrame,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
    y_use_base_only: bool = False,
):
    """(T,N) -> x:(B,L,N,D), y:(B,H,N,D or 1)  [float32]"""
    assert isinstance(df.index, pd.DatetimeIndex), "df.index must be DatetimeIndex"
    T, N = df.shape

    base_vals = df.values.astype(np.float32)            # (T,N)
    base_vals = np.expand_dims(base_vals, axis=-1)      # (T,N,1)
    data_list = [base_vals]

    if add_time_in_day:
        tod = ((df.index.values - df.index.values.astype("datetime64[D]"))
               / np.timedelta64(1, "D")).astype(np.float32)      # (T,)
        tod = np.repeat(tod[:, None, None], N, axis=1)           # (T,N,1)
        data_list.append(tod)

    if add_day_in_week:
        dow = df.index.dayofweek.values                           # (T,)
        onehot = np.eye(7, dtype=np.float32)[dow]                # (T,7)
        onehot = np.repeat(onehot[:, None, :], N, axis=1)        # (T,N,7)
        data_list.append(onehot)

    data = np.concatenate(data_list, axis=-1).astype(np.float32) # (T,N,D)

    xs, ys = [], []
    min_t = abs(int(np.min(x_offsets)))
    max_t = int(T - np.max(y_offsets))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]     # (L,N,D)
        y_t = data[t + y_offsets, ...]     # (H,N,D)
        if y_use_base_only:
            y_t = y_t[..., [0]]            # (H,N,1)
        xs.append(x_t)
        ys.append(y_t)

    x = np.stack(xs, 0).astype(np.float32)
    y = np.stack(ys, 0).astype(np.float32)
    return x, y

def save_npz(out_dir: str, x: np.ndarray, y: np.ndarray,
             x_offsets: np.ndarray, y_offsets: np.ndarray,
             var_cols: list[str], train_ratio=0.7, test_ratio=0.2):
    os.makedirs(out_dir, exist_ok=True)
    B = x.shape[0]
    n_test = int(round(B * test_ratio))
    n_train = int(round(B * train_ratio))
    n_val = B - n_test - n_train

    splits = {
        "train": (x[:n_train], y[:n_train]),
        "val":   (x[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        "test":  (x[-n_test:], y[-n_test:]),
    }
    for name, (xx, yy) in splits.items():
        np.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"),
            x=xx.astype(np.float32, copy=False),
            y=yy.astype(np.float32, copy=False),
            x_offsets=x_offsets.reshape(-1,1).astype(np.int64),
            y_offsets=y_offsets.reshape(-1,1).astype(np.int64),
            var_columns=np.array(var_cols, dtype=object),
        )
        print(f"[{name}] x{xx.shape}, y{yy.shape} -> {os.path.join(out_dir, f'{name}.npz')}")

# --------------------------
# 메인 파이프라인
# --------------------------
def process_one(csv_path: str, out_dir: str,
                var_columns, seq_in_len, seq_out_len,
                add_time_in_day, add_day_in_week,
                train_ratio, test_ratio,
                interpolate, hour_policy,
                y_use_base_only):
    print(f"\n=== Processing: {csv_path} ===")
    df = read_csv_to_frame(
        csv_path,
        var_columns=var_columns,
        interpolate=interpolate,
        hour_policy=hour_policy,
    )
    x_offsets = np.arange(-seq_in_len+1, 1, 1, dtype=np.int64)  # [-L+1,...,0]
    y_offsets = np.arange(1, seq_out_len+1, 1, dtype=np.int64)  # [1,...,H]

    x, y = generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
        y_use_base_only=y_use_base_only,
    )
    print(f"x shape: {x.shape} | y shape: {y.shape}")
    save_npz(out_dir, x, y, x_offsets, y_offsets, var_columns, train_ratio, test_ratio)

    # --- 빠른 검증(샘플 1개) ---
    try:
        print("\n[Quick sanity check: x[0], y[0] vs CSV 상위 행]")
        L, H = len(x_offsets), len(y_offsets)
        # 원본 CSV를 그대로 로드 (일자/시간/변수만)
        raw = pd.read_csv(csv_path)
        cols = ["day", "hour_t"] + var_columns
        print(raw[cols].head(L + H))

        x0 = x[0][..., 0]  # (L,N) 원자료 채널
        y0 = y[0][..., 0]  # (H,N) 원자료 채널(또는 1)
        print("\nnpz x[0][:,:,0] (입력 L행):\n", np.round(x0, 6))
        print("\nnpz y[0][:,:,0] (타깃 H행):\n", np.round(y0, 6))
    except Exception as e:
        print("[Sanity check skipped]", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="file_num_*.csv 위치 폴더")
    ap.add_argument("--output_base", type=str, required=True, help="npz 저장 루트 (예: ./data/MYDATA)")
    ap.add_argument("--start_id", type=int, default=2, help="파일 시작 id (file_num_{id}.csv)")
    ap.add_argument("--end_id", type=int, default=None, help="파일 끝 id (미지정 시 start_id 이상 전부)")
    ap.add_argument("--seq_in_len", type=int, default=12)
    ap.add_argument("--seq_out_len", type=int, default=12)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--no_time_in_day", action="store_true", help="일중 시간 특성 제외")
    ap.add_argument("--day_of_week", action="store_true", help="요일 one-hot(7) 포함")
    ap.add_argument("--var_columns", type=str, default=",".join(VAR_COLUMNS_DEFAULT))
    ap.add_argument("--no_interpolate", action="store_true", help="시간 보간 끄기")
    ap.add_argument("--hour_policy", type=str, default="round", choices=["floor","round","ceil"])
    ap.add_argument("--y_use_base_only", action="store_true", help="y를 원자료 채널 1개로만")
    args = ap.parse_args()

    var_cols = [c.strip() for c in args.var_columns.split(",")]

    paths = sorted(glob.glob(os.path.join(args.input_dir, "file_num_*.csv")))
    found = []
    for p in paths:
        m = re.search(r"file_num_(\d+)\.csv$", os.path.basename(p))
        if m:
            pid = int(m.group(1))
            found.append((pid, p))
    found.sort(key=lambda x: x[0])

    if args.end_id is not None:
        found = [(pid, p) for pid, p in found if args.start_id <= pid <= args.end_id]
    else:
        found = [(pid, p) for pid, p in found if pid >= args.start_id]

    if not found:
        print("No matching files.")
        return

    print(f"Found {len(found)} participants:", [pid for pid, _ in found])

    for pid, csv_path in found:
        out_dir = os.path.join(args.output_base, f"data_{pid}")
        process_one(
            csv_path=csv_path,
            out_dir=out_dir,
            var_columns=var_cols,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            add_time_in_day=(not args.no_time_in_day),
            add_day_in_week=args.day_of_week,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            interpolate=(not args.no_interpolate),
            hour_policy=args.hour_policy,
            y_use_base_only=args.y_use_base_only,
        )

if __name__ == "__main__":
    main()