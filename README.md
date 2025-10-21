# Dtx_MTGNN
- Dtx 프로젝트 데이터 분석
- MTGNN 모델 응용에 목적
- 설명
  1. MTGNN 깃허브의 net.py, layer.py 모델 구조는 유지 (train_multi_step.py 변형)
  2. 입력 데이터 Sequence 유지(argparser에서 유지됨)
  3. 우리가 가진 csv 데이터를 npz 형식으로 변환, 모델에 입력(convert_and_check.py 새롭게 생성)
  4. 결과 메트릭(MAE, RMSE, MAPE) 및 실제 값과 비교 출력(evaluate_mtgnn.py)

- 예시데이터 및 다른 모델과 비교 가능((실제, 예시) x (GNN, LSTM))
- 
