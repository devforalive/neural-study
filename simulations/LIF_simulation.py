# [파이썬 시뮬레이션 1] LIF 모델 활동전위 시뮬레이션
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
tau_m    = 20e-3    # 막 시정수 (20 ms)
V_rest   = -70e-3   # 안정막전위 (-70 mV)
V_thresh = -55e-3   # 역치 (-55 mV)
V_reset  = -75e-3   # 리셋 전위 (-75 mV)
R_m      = 10e6     # 막 저항 (10 MΩ)

dt = 0.05e-3        # 시간 간격 (0.05 ms)
T  = 150e-3         # 총 시간 (150 ms)
t  = np.arange(0, T, dt)
V  = np.full(len(t), V_rest)

# 입력 전류: 40~120 ms 구간에만 전류 주입
I = np.zeros(len(t))
I[(t >= 0.04) & (t <= 0.12)] = 2.8e-9

# 매 시간 간격마다 막전위 업데이트
for i in range(1, len(t)):
    dV = (-(V[i-1] - V_rest) + R_m * I[i]) / tau_m
    V[i] = V[i-1] + dV * dt
    if V[i] >= V_thresh:   # 역치 도달 → 스파이크
        V[i] = +40e-3
        if i+1 < len(t):
            V[i+1] = V_reset
