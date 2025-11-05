# Dtx_MTGNN
- Dtx 프로젝트 EMA 데이터 분석
- "dstot", "eu_lag_sum", "hr_mean", "hr_var", "pain_mean" : 다섯 개 변인으로 구성됨
- "pain_mean" 예측(분류)을 목적으로 함
- 본 페이지에서는 MTGNN 모델 적용
- 예시데이터 및 다른 모델과 비교 가능((실제, 예시) x (GNN, LSTM))

  ```
  Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020, August). Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 753-763).)

  ```
  
## 코드 설명
- 설명
  1. train_multi_step.py : MTGNN 원문에서 제공된 net.py, layer.py 등 모델 구조 유지
  2. convert_and_check.py : 우리가 가진 csv 데이터를 npz 형식으로 변환, 모델에 입력
  3. evaluate_mtgnn.py : 결과 메트릭(MAE, RMSE, MAPE) 출력
  4. compare_test_table.py : validation 데이터 배치 중 하나 선택, 실제값-예측값 시각화

## requirements
```
pip install -r requirements.txt
```


## 실행 커맨드
### 0. 공통
- (학습 데이터 생성)
```
python3 convert_and_check.py \
  --input_dir ./data/split \
  --output_base ./data/MYDATA \
  --start_id 2 --end_id 50 \
  --seq_in_len 12 --seq_out_len 12 \
  --no_time_in_day \
  --hour_policy round \
  --no_interpolate \
  --y_use_base_only
```
### 1. MTGNN
- (학습)
```
python3 train_multi_step.py \
  --data ./data/MYDATA/data_7 \
  --num_nodes 5 \
  --in_dim 1 \
  --seq_in_len 12 \
  --seq_out_len 12 \
  --gcn_true true \
  --buildA_true true \
  --adj_data "" \
  --subgraph_size 4 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --epochs 50 \
  --save ./save/data_38 \
  --expid 38
```
- (평가)
```
python3 evaluate_mtgnn.py \
--data_dir ./data/MYDATA/data_7 \
--ckpt ./save/data_7/exp7_2.pth \
--num_nodes 5 \
--in_dim 1 \
--seq_in_len 12 \
--seq_out_len 12 \
--subgraph_size 4 \
--split val \
--batch_index 0 \
--save_csv ./evaluate/evaluate_GNN_b0_tidy_38.csv \
--save_wide_csv ./evaluate_GNN_b0_wide_38.csv
```
- (실제 값과 비교)
```
python3 compare_test_table.py \
  --data_dir ./data/MYDATA/data_7 \
  --ckpt ./save/data_7/exp7_2.pth \
  --num_nodes 5 \
  --in_dim 1 \
  --seq_in_len 12 \
  --seq_out_len 12 \
  --save_csv ./compare/test_true_pred_table_7.csv \
  --limit_rows 100
```



## 자료 구조
```
MTGNN/
├─ train_multi_step.py
├─ replay_mtgnn.py
├─ compare_test_table.py
├─ compare_test_table_lstm.py
├─ util.py
├─ trainer.py
├─ net.py
│
├─ data/
│  ├─ MYDATA/
│  │  ├─ data_1/
│  │  │  ├─ train.npz          # x:(B,L,N,C), y:(B,H,N,C), (옵션)var_columns
│  │  │  ├─ val.npz            # 동일 포맷
│  │  │  └─ test.npz           # 동일 포맷
│  │  ├─ data_2/
│  │  │  ├─ train.npz
│  │  │  ├─ val.npz
│  │  │  └─ test.npz
│  │  └─ … (data_3 ~ data_50)
│  │
│  ├─ MYDATA_sample/
│  │  └─ data_1/
│  │     ├─ train.npz
│  │     ├─ val.npz
│  │     └─ test.npz
│  │
│  ├─ sensor_graph/
│  │  └─ adj_mx.pkl            # (옵션) 사전 인접행렬
│  │
│  └─ split/                   # 합성/원천 CSV (make_easy_synth_split.py 결과)
│     ├─ file_num_2.csv
│     ├─ file_num_3.csv
│     └─ …
│
├─ save/                       # MTGNN 학습 결과
│  ├─ data_1/
│  │  ├─ exp1_0.pth
│  │  ├─ exp1_1.pth
│  │  ├─ …
│  │  ├─ best_exp1.pth         # (베스트 링크/복사본 만들었을 때)
│  │  ├─ exp1_0_adj.npy        # 적응형 인접행렬 저장본
│  │  ├─ exp1_0_adj.png        # 적응형 인접행렬 히트맵
│  │  └─ …
│  ├─ data_2/
│  │  └─ …
│  └─ …
│
├─ save_sample/                # 샘플 데이터 학습 결과
│  └─ data_1/
│     ├─ exp1_0.pth
│     └─ …
│
```
