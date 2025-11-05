# replay_mtgnn.py (enhanced: per-variable comparison + tidy/wide CSV)
# 사용 예:
# python3 replay_mtgnn.py \
#   --data_dir ./data/MYDATA/data_2 \
#   --ckpt ./save/data_2/exp1_9.pth \
#   --num_nodes 5 --in_dim 2 --seq_in_len 12 --seq_out_len 12 \
#   --subgraph_size 4 \
#   --split val --batch_index 0 \
#   --save_csv ./replay_val_b0_tidy.csv \
#   --save_wide_csv ./replay_val_b0_wide.csv

import os
import argparse
import numpy as np
import torch
import pandas as pd

from util import load_dataset, metric
from net import gtnet

def get_one_batch(dataloader_split, batch_index=0):
    """커스텀 로더(get_iterator)에서 batch_index번째 배치를 반환"""
    for i, (x, y) in enumerate(dataloader_split.get_iterator()):
        if i == batch_index:
            return x, y
    raise IndexError(f"batch_index {batch_index} out of range.")

def build_model(args, device):
    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=args.num_nodes,
        device=device,
        predefined_A=None,          # adaptive adjacency만
        dropout=0.3,
        subgraph_size=min(args.subgraph_size, args.num_nodes),
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=args.seq_in_len,
        in_dim=args.in_dim,
        out_dim=args.seq_out_len,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True
    ).to(device)
    return model

def safe_varnames(data_dir: str, split: str, num_nodes: int):
    """split npz에서 var_columns를 읽어 노드명 리스트 반환. 실패시 기본 이름."""
    npz_path = os.path.join(data_dir, f"{split}.npz")
    names = [f"node_{i}" for i in range(num_nodes)]
    try:
        if os.path.exists(npz_path):
            z = np.load(npz_path, allow_pickle=True)
            if "var_columns" in z:
                cols = [str(c) for c in z["var_columns"].tolist()]
                if len(cols) == num_nodes:
                    names = cols
    except Exception:
        pass
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="train/val/test.npz 폴더")
    ap.add_argument("--ckpt", type=str, required=True, help="학습된 .pth 경로")
    ap.add_argument("--num_nodes", type=int, required=True)
    ap.add_argument("--in_dim", type=int, default=2)
    ap.add_argument("--seq_in_len", type=int, default=12)
    ap.add_argument("--seq_out_len", type=int, default=12)
    ap.add_argument("--subgraph_size", type=int, default=4)
    ap.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    ap.add_argument("--batch_index", type=int, default=0)
    ap.add_argument("--save_csv", type=str, default="", help="Tidy 포맷 저장 경로")
    ap.add_argument("--save_wide_csv", type=str, default="", help="Wide 포맷 저장 경로")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터 로드
    dl = load_dataset(args.data_dir, 64, 64, 64)
    scaler = dl["scaler"]
    if args.split == "train":
        split_loader = dl["train_loader"]
    elif args.split == "val":
        split_loader = dl["val_loader"]
    else:
        split_loader = dl["test_loader"]

    # 2) 모델 구성 + 체크포인트 로드
    model = build_model(args, device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[OK] checkpoint loaded:", args.ckpt)

    # 3) 배치 하나 가져오기
    x_np, y_np = get_one_batch(split_loader, batch_index=args.batch_index)  # x:(B,L,N,F), y:(B,L,N,F)
    # dtype 안전
    x = torch.tensor(x_np, device=device).float().transpose(1, 3)  # -> (B,F,N,L)
    y = torch.tensor(y_np, device=device).float().transpose(1, 3)  # -> (B,F,N,H)
    y_target = y[:, 0, :, :]  # (B,N,H) : 첫 채널이 원자료

    # 4) 예측
    with torch.no_grad():
        pred = model(x)             # (B, 1, N, H)  (MTGNN 구현에 따라 다를 수 있음)
        # pred를 (B,N,H)로
        if pred.dim() == 4:         # (B, C=1, N, H)
            pred = pred[:, 0, :, :] # (B,N,H)
        elif pred.dim() == 3:       # 이미 (B,N,H)
            pass
        else:
            raise RuntimeError(f"Unexpected pred shape: {tuple(pred.shape)}")

    # 5) 역스케일링
    # scaler.inverse_transform은 (B,N,H) 텐서도 처리하도록 구현되어 있음(원 레포 기준)
    pred_denorm = scaler.inverse_transform(pred)
    real_denorm = scaler.inverse_transform(y_target)

    # 전체 지표
    mae, mape, rmse = metric(pred_denorm, real_denorm)
    print(f"[{args.split.upper()} batch {args.batch_index}]  MAE={mae:.4f}  MAPE={mape:.4f}  RMSE={rmse:.4f}")

    # 6) 변수명 로딩
    var_names = safe_varnames(args.data_dir, args.split, args.num_nodes)

    # 7) B=0 샘플에 대해 변수별, 수평선별 비교표 생성
    b = 0
    B, N, H = pred_denorm.shape
    y_true_np = real_denorm[b].detach().cpu().numpy()  # (N,H)
    y_pred_np = pred_denorm[b].detach().cpu().numpy()  # (N,H)

    # Tidy 포맷
    rows = []
    for n in range(N):
        for h in range(H):
            yt = float(y_true_np[n, h])
            yp = float(y_pred_np[n, h])
            ae = abs(yp - yt)
            pe = (ae / (abs(yt) + 1e-8)) * 100.0
            rows.append({
                "variable": var_names[n],
                "node": n,
                "horizon": h+1,
                "y_true": yt,
                "y_pred": yp,
                "abs_err": ae,
                "pct_err": pe
            })
    df_tidy = pd.DataFrame(rows)

    # Wide 포맷: 변수 한 줄에 h1_true,h1_pred,...,hH_true,hH_pred
    wide_cols = {}
    for n in range(N):
        for h in range(H):
            wide_cols.setdefault("variable", []).append(var_names[n] if h==0 else None)
        # 위 방식은 보기 어렵다 → 변수 단위로 행을 하나씩 만들자
    wide_rows = []
    for n in range(N):
        rec = {"variable": var_names[n], "node": n}
        for h in range(H):
            rec[f"h{h+1}_true"] = float(y_true_np[n, h])
            rec[f"h{h+1}_pred"] = float(y_pred_np[n, h])
        wide_rows.append(rec)
    df_wide = pd.DataFrame(wide_rows)

    # 8) 변수별 요약(MAE/RMSE) 표 출력
    per_var = (
        df_tidy.groupby(["variable","node"])
        .apply(lambda g: pd.Series({
            "MAE":  np.mean(np.abs(g["y_pred"] - g["y_true"])),
            "RMSE": np.sqrt(np.mean((g["y_pred"] - g["y_true"])**2)),
            # ▼ 추가: MAPE (비율)과 MAPE(%) 둘 다 보고 싶다면 둘 다 넣기
            "MAPE":  np.mean(np.abs(g["y_pred"] - g["y_true"]) / (np.abs(g["y_true"]) + 1e-8)),
            "MAPE_%": np.mean(g["pct_err"]),  # 이미 %로 계산한 값의 평균
        }))
        .reset_index()
        .sort_values("MAE")
    )
    print("\n[Per-variable summary on selected sample (B=0)]")
    print(per_var.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 9) 콘솔 미리보기: 첫 변수 1개 예시
    if N > 0:
        print("\n[Preview] variable:", var_names[0])
        print("True:", np.round(y_true_np[0, :], 4))
        print("Pred:", np.round(y_pred_np[0, :], 4))

    # 10) CSV 저장 (옵션)
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        df_tidy.to_csv(args.save_csv, index=False)
        print("[Saved tidy]", args.save_csv)
    if args.save_wide_csv:
        os.makedirs(os.path.dirname(args.save_wide_csv) or ".", exist_ok=True)
        df_wide.to_csv(args.save_wide_csv, index=False)
        print("[Saved wide]", args.save_wide_csv)

if __name__ == "__main__":
    main()