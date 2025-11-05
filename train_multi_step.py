import os
import time
import argparse
import numpy as np
import torch

from util import *
from trainer import Trainer
from net import gtnet


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')

parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')

parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')

parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save dir (folder)')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

parser.add_argument('--runs', type=int, default=10, help='number of runs')

# ▶︎ 추가: 0 값이 있는 타깃을 위한 안전 MAPE 설정
parser.add_argument('--mape_eps', type=float, default=1e-6, help='epsilon for safe MAPE denominator')
parser.add_argument('--mape_ignore_zeros', type=str_to_bool, default=True, help='if True, ignore |y| < eps when computing MAPE')

args = parser.parse_args()
torch.set_num_threads(3)


def safe_metrics(pred, true, eps=1e-6, ignore_zeros=True):
    """
    pred, true: shape (...), torch.Tensor or numpy.ndarray
    Returns: (MAE, MAPE, RMSE) as python floats
    - MAPE: if ignore_zeros=True, only averages over |true| >= eps
            else uses max(|true|, eps) in denominator
    """
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor): true = true.detach().cpu().numpy()

    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = np.abs(true)
    if ignore_zeros:
        mask = denom >= eps
        if mask.any():
            mape = float(np.mean(np.abs(err[mask]) / denom[mask]) * 100.0)
        else:
            mape = float('nan')  # 전부 0인 경우
    else:
        mape = float(np.mean(np.abs(err) / np.maximum(denom, eps)) * 100.0)

    return mae, mape, rmse


def main(runid):
    # 저장 폴더 보장
    os.makedirs(args.save, exist_ok=True)

    # 디바이스 & 데이터
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # 사전 인접행렬(Optional)
    predefined_A = None
    if args.adj_data and os.path.exists(args.adj_data):
        A = load_adj(args.adj_data)
        predefined_A = torch.tensor(A, dtype=torch.float32, device=device)
        if predefined_A.shape[0] == args.num_nodes:
            predefined_A = predefined_A - torch.eye(args.num_nodes, device=device)
        print(f"[Info] Loaded predefined adjacency from {args.adj_data}")
    else:
        print("[Info] No predefined adjacency. Using adaptive adjacency only.")

    # 모델 (회귀용 그대로)
    model = gtnet(
        args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
        device, predefined_A=predefined_A,
        dropout=args.dropout, subgraph_size=args.subgraph_size,
        node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
        skip_channels=args.skip_channels, end_channels=args.end_channels,
        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
        layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True
    )

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device, args.cl)

    print("start training...", flush=True)
    his_loss, val_time, train_time = [], [], []
    minl = 1e5

    # 체크포인트 경로
    best_ckpt = os.path.join(args.save, f"exp{args.expid}_{runid}.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()

        dataloader['train_loader'].shuffle()
        for it, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.tensor(x, device=device).transpose(1, 3)   # (B,C,N,L) ← (B,L,N,C)
            trainy = torch.tensor(y, device=device).transpose(1, 3)   # (B,C,N,H)

            if it % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)

            for j in range(args.num_split):
                if j != args.num_split - 1:
                    ids = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    ids = perm[j * num_sub:]
                ids = torch.tensor(ids, device=device)
                tx = trainx[:, :, ids, :]
                ty = trainy[:, :, ids, :]
                # 회귀: 첫 채널만 예측 (원래 코드 유지)
                metrics = engine.train(tx, ty[:, 0, :, :], ids)
                train_loss.append(metrics[0]); train_mape.append(metrics[1]); train_rmse.append(metrics[2])

            if it % args.print_every == 0 and len(train_loss) > 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(it, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation (안전 MAPE로만 대체하지 않고, engine.eval은 그대로 호출 + 별도 안전지표도 계산)
        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for it, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            vx = torch.tensor(x, device=device).transpose(1, 3)
            vy = torch.tensor(y, device=device).transpose(1, 3)
            metrics = engine.eval(vx, vy[:, 0, :, :])  # 엔진 내부 지표 유지
            valid_loss.append(metrics[0]); valid_mape.append(metrics[1]); valid_rmse.append(metrics[2])
        s2 = time.time()
        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = float(np.mean(train_loss)); mtrain_mape = float(np.mean(train_mape)); mtrain_rmse = float(np.mean(train_rmse))
        mvalid_loss = float(np.mean(valid_loss)); mvalid_mape = float(np.mean(valid_mape)); mvalid_rmse = float(np.mean(valid_rmse))
        his_loss.append(mvalid_loss)

        log = ('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
               'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch')
        print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), best_ckpt)
            print(f"[Saved] checkpoint -> {best_ckpt}")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # 베스트 로드
    if os.path.exists(best_ckpt):
        engine.model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"[Loaded] best checkpoint -> {best_ckpt}")
    else:
        print(f"[Warn] no checkpoint found at {best_ckpt}. Using current model for eval.")

    print("Training finished")
    bestid = int(np.argmin(his_loss))
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # ---------- 학습된 적응형 인접행렬 저장 ----------
    with torch.no_grad():
        adj = None
        try:
            idx = torch.arange(args.num_nodes, device=device)
            if hasattr(engine.model, 'gc'):
                adj = engine.model.gc(idx).detach().cpu().numpy()
        except Exception:
            pass
        if adj is None:
            if hasattr(engine.model, 'adaptive_adj'):
                adj = engine.model.adaptive_adj.detach().cpu().numpy()
            elif hasattr(engine.model, 'adp'):
                adj = engine.model.adp.detach().cpu().numpy()

    if adj is not None:
        npy_path = os.path.join(args.save, f"exp{args.expid}_{runid}_adj.npy")
        png_path = os.path.join(args.save, f"exp{args.expid}_{runid}_adj.png")
        np.save(npy_path, adj)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(adj, cmap="viridis")
        plt.title("Learned Adaptive Adjacency Matrix")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"[Saved] learned adjacency -> {npy_path} & {png_path}")
    else:
        print("[Warning] No adaptive adjacency attribute/method found in model")

    # ---------- 평가 (안전 MAPE) ----------
    # valid
    outputs = []
    realy = torch.tensor(dataloader['y_val'], device=device).transpose(1, 3)[:, 0, :, :]  # (S,N,H)
    for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        vx = torch.tensor(x, device=device).transpose(1, 3)  # (B,C,N,L)
        with torch.no_grad():
            preds = engine.model(vx)  # (B, C_out=1, N, H) in this config
            if preds.dim() == 4:
                preds = preds[:, 0, :, :]  # (B,N,H)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]  # (S,N,H)
    pred = scaler.inverse_transform(yhat)
    # 안전 지표
    vmae, vmape, vrmse = safe_metrics(pred, realy, eps=args.mape_eps, ignore_zeros=args.mape_ignore_zeros)
    print(f"[VALID-safe] MAE={vmae:.4f}  MAPE={vmape:.4f}  RMSE={vrmse:.4f}")

    # test
    outputs = []
    realy = torch.tensor(dataloader['y_test'], device=device).transpose(1, 3)[:, 0, :, :]  # (S,N,H)
    for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        tx = torch.tensor(x, device=device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(tx)  # (B,1,N,H)
            if preds.dim() == 4:
                preds = preds[:, 0, :, :]  # (B,N,H)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]  # (S,N,H)

    mae, mape, rmse = [], [], []
    for h in range(args.seq_out_len):
        ph = scaler.inverse_transform(yhat[:, :, h])  # (S,N)
        rh = realy[:, :, h]                           # (S,N)
        m = safe_metrics(ph, rh, eps=args.mape_eps, ignore_zeros=args.mape_ignore_zeros)
        print('Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
              .format(h + 1, m[0], m[1], m[2]))
        mae.append(m[0]); mape.append(m[1]); rmse.append(m[2])

    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == "__main__":
    vmae, vmape, vrmse = [], [], []
    mae, mape, rmse = [], [], []

    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1); vmape.append(vm2); vrmse.append(vm3)
        mae.append(m1); mape.append(m2); rmse.append(m3)

    mae = np.array(mae); mape = np.array(mape); rmse = np.array(rmse)
    amae = np.mean(mae, 0); amape = np.mean(mape, 0); armse = np.mean(rmse, 0)
    smae = np.std(mae, 0);  smape = np.std(mape, 0);  srmse = np.std(rmse, 0)

    print('\n\nResults for {} runs\n'.format(args.runs))
    print('valid\tMAE\tRMSE\tMAPE')
    print('mean:\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    print('std:\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n')
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in range(len(amae)):
        print('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
            .format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))