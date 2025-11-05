# import torch.optim as optim
# import math
# from net import *
# import util
# class Trainer():
#     def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
#         self.scaler = scaler
#         self.model = model
#         self.model.to(device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
#         self.loss = util.masked_mae
#         self.clip = clip
#         self.step = step_size
#         self.iter = 1
#         self.task_level = 1
#         self.seq_out_len = seq_out_len
#         self.cl = cl

#     # def train(self, input, real_val, idx=None):
#     #     self.model.train()
#     #     self.optimizer.zero_grad()
#     #     output = self.model(input, idx=idx)
#     #     output = output.transpose(1,3)
#     #     real = torch.unsqueeze(real_val,dim=1)
#     #     predict = self.scaler.inverse_transform(output)
#     #     if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
#     #         self.task_level +=1
#     #     if self.cl:
#     #         loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
#     #     else:
#     #         loss = self.loss(predict, real, 0.0)

#     #     loss.backward()

#     #     if self.clip is not None:
#     #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

#     #     self.optimizer.step()
#     #     # mae = util.masked_mae(predict,real,0.0).item()
#     #     mape = util.masked_mape(predict,real,0.0).item()
#     #     rmse = util.masked_rmse(predict,real,0.0).item()
#     #     self.iter += 1
#     #     return loss.item(),mape,rmse
    
    
#     def train(self, x, y, ids=None):
#         self.model.train()                       # ← 반드시 train 모드
#         self.optimizer.zero_grad(set_to_none=True)

#         predict = self.model(x)                  # ← no_grad 금지, detach 금지
#         # MTGNN 출력이 (B,1,N,H)일 수도 있으니 필요한 최소한의 reshape만 텐서 상태로
#         if predict.dim() == 4:                   # (B, C=1, N, H)
#             predict = predict[:, 0, :, :]        # (B, N, H)

#         # y는 라벨이므로 grad 불필요. 하지만 torch 텐서여야 함 (numpy 금지)
#         # 손실: util.py의 masked_* 함수는 torch 텐서를 반환(OK)
#         loss = self.loss(predict, y, 0.0)

#         # 디버깅: 그래프 유효성 체크
#         assert predict.requires_grad, "predict has no grad (was it detached or computed under no_grad?)"
#         assert loss.requires_grad, "loss has no grad (was it converted to float/item()?)"

#         loss.backward()
#         if self.clip is not None and self.clip > 0:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
#         self.optimizer.step()

#         # 로그용으로만 .item()
#         mae = self.mae_fn(predict, y).item()
#         mape = self.mape_fn(predict, y).item()
#         rmse = self.rmse_fn(predict, y).item()
#         return loss.item(), mape, rmse

#     def eval(self, input, real_val):
#         self.model.eval()
#         output = self.model(input)
#         output = output.transpose(1,3)
#         real = torch.unsqueeze(real_val,dim=1)
#         predict = self.scaler.inverse_transform(output)
#         loss = self.loss(predict, real, 0.0)
#         mape = util.masked_mape(predict,real,0.0).item()
#         rmse = util.masked_rmse(predict,real,0.0).item()
#         return loss.item(),mape,rmse



# class Optim(object):

#     def _makeOptimizer(self):
#         if self.method == 'sgd':
#             self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adagrad':
#             self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adadelta':
#             self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         elif self.method == 'adam':
#             self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
#         else:
#             raise RuntimeError("Invalid optim method: " + self.method)

#     def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
#         self.params = params  # careful: params may be a generator
#         self.last_ppl = None
#         self.lr = lr
#         self.clip = clip
#         self.method = method
#         self.lr_decay = lr_decay
#         self.start_decay_at = start_decay_at
#         self.start_decay = False

#         self._makeOptimizer()

#     def step(self):
#         # Compute gradients norm.
#         grad_norm = 0
#         if self.clip is not None:
#             torch.nn.utils.clip_grad_norm_(self.params, self.clip)

#         # for param in self.params:
#         #     grad_norm += math.pow(param.grad.data.norm(), 2)
#         #
#         # grad_norm = math.sqrt(grad_norm)
#         # if grad_norm > 0:
#         #     shrinkage = self.max_grad_norm / grad_norm
#         # else:
#         #     shrinkage = 1.
#         #
#         # for param in self.params:
#         #     if shrinkage < 1:
#         #         param.grad.data.mul_(shrinkage)
#         self.optimizer.step()
#         return  grad_norm

#     # decay learning rate if val perf does not improve or we hit the start_decay_at limit
#     def updateLearningRate(self, ppl, epoch):
#         if self.start_decay_at is not None and epoch >= self.start_decay_at:
#             self.start_decay = True
#         if self.last_ppl is not None and ppl > self.last_ppl:
#             self.start_decay = True

#         if self.start_decay:
#             self.lr = self.lr * self.lr_decay
#             print("Decaying learning rate to %g" % self.lr)
#         #only decay for one epoch
#         self.start_decay = False

#         self.last_ppl = ppl

#         self._makeOptimizer()

# trainer.py
import torch
import torch.optim as optim
from util import masked_mae, masked_mape, masked_rmse

class Trainer:
    def __init__(self, model, lrate, wdecay, clip, step_size,
                 seq_out_len, scaler, device, cl=True):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        # 손실/지표 함수 바인딩 (항상 torch.Tensor를 반환)
        self.loss_fn = masked_mae
        self.mae_fn  = masked_mae
        self.mape_fn = masked_mape
        self.rmse_fn = masked_rmse

        self.scaler = scaler                # 역스케일링은 리포팅 때만 사용
        self.clip = clip
        self.step = step_size               # 커리큘럼 증가 주기
        self.iter = 1
        self.task_level = 1                 # 예: 처음엔 horizon 1만 학습
        self.seq_out_len = seq_out_len
        self.cl = cl                        # 커리큘럼 학습 on/off

    def _forward_predict(self, x, ids=None):
        """
        모델 출력을 (B,N,H)로 정규화.
        gtnet이 (B,1,N,H) 또는 (B,N,H)를 반환할 수 있으니 안전하게 처리.
        """
        if ids is None:
            out = self.model(x)
        else:
            out = self.model(x, idx=ids)

        # 다양한 구현 호환
        if out.dim() == 4:          # (B, C, N, H)
            if out.size(1) == 1:    # C=1
                out = out[:, 0, :, :]     # -> (B,N,H)
            else:
                # 필요 시 첫 채널만 사용
                out = out[:, 0, :, :]
        elif out.dim() == 3:        # (B,N,H)
            pass
        else:
            raise RuntimeError(f"Unexpected prediction shape: {tuple(out.shape)}")
        return out

    def train(self, x, y, ids=None):
        """
        x: (B, F, N, L)  — main에서 이미 .transpose(1,3) 등 맞춰서 들어옴
        y: (B, N, H)     — main에서 y = trainy[:, 0, :, :] 형태로 전달
        ids: (optional)  — 부분 그래프 학습 시 서브셋 인덱스
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pred = self._forward_predict(x, ids=ids)  # (B,N,H)

        # 커리큘럼 학습: 현재 horizon까지만 손실 계산
        if self.cl:
            H = min(self.task_level, self.seq_out_len)
            loss = self.loss_fn(pred[:, :, :H], y[:, :, :H], 0.0)
        else:
            loss = self.loss_fn(pred, y, 0.0)

        # 역전파
        loss.backward()
        if self.clip is not None and self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # task_level 갱신
        if (self.iter % self.step == 0) and (self.task_level < self.seq_out_len):
            self.task_level += 1
        self.iter += 1

        # 로그용 지표(표준화 공간)
        mae  = self.mae_fn(pred,  y, 0.0).item()
        mape = self.mape_fn(pred, y, 0.0).item()
        rmse = self.rmse_fn(pred, y, 0.0).item()
        return loss.item(), mape, rmse

    @torch.no_grad()
    def eval(self, x, y):
        """
        x: (B, F, N, L)
        y: (B, N, H)
        검증도 학습과 동일하게 '표준화된 공간'에서 손실 계산 (일관성 유지)
        """
        self.model.eval()
        pred = self._forward_predict(x, ids=None)  # (B,N,H)

        loss = self.loss_fn(pred, y, 0.0)
        mae  = self.mae_fn(pred,  y, 0.0).item()
        mape = self.mape_fn(pred, y, 0.0).item()
        rmse = self.rmse_fn(pred, y, 0.0).item()
        return loss.item(), mape, rmse