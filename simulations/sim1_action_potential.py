import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
def sim1_action_potential():
    rcParams['font.family'] = 'DejaVu Sans'
 
    # ── 파라미터 ──
    tau_m    = 20e-3
    V_rest   = -70e-3
    V_thresh = -55e-3
    V_reset  = -75e-3
    R_m      = 10e6
    
    dt = 0.05e-3
    T  = 150e-3
    t  = np.arange(0, T, dt)
    V  = np.full(len(t), V_rest)
    
    I = np.zeros(len(t))
    I[(t >= 0.04) & (t <= 0.12)] = 2.8e-9
    
    # ── 스파이크 파형 템플릿 (실제 활동전위 모양) ──
    # 시간축(ms): 역치(-55) → 피크(+40) → 재분극(0) → 과분극(-75) → 안정(-70)
    spike_t_ms = np.array([0,   0.3, 0.8,  1.4,   2.2,   3.5,   5.0 ])
    spike_v    = np.array([-55,  40,   5,  -20,   -75,   -71,   -70 ]) * 1e-3
    
    # ms → 인덱스 수
    spike_len = int(5.0e-3 / dt)  # 5ms 분량
    spike_idx = np.arange(spike_len)
    spike_t_interp = spike_idx * dt * 1000  # ms 단위
    
    # 템플릿을 인덱스 배열로 보간
    spike_template = np.interp(spike_t_interp, spike_t_ms, spike_v)
    
    # 시뮬레이션 
    skip_until = 0   # 스파이크 파형 삽입 중엔 LIF 적분 건너뜀
    spikes = []
    
    for i in range(1, len(t)):
        if i < skip_until:
            continue  # 파형 삽입 구간은 이미 덮어씀
    
        dV = (-(V[i-1] - V_rest) + R_m * I[i]) / tau_m
        V[i] = V[i-1] + dV * dt
    
        if V[i] >= V_thresh:
            spikes.append(i)
            # 스파이크 파형 삽입
            end = min(i + spike_len, len(t))
            V[i:end] = spike_template[:end - i]
            skip_until = end  # 파형 구간 건너뜀
    
    print(f"총 스파이크 수: {len(spikes)}")
    print(f"평균 발화율: {len(spikes)/0.08:.1f} Hz")
    
    # ── 그래프 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white')
    
    ax1.plot(t * 1000, V * 1000, color='#2563EB', linewidth=1.8,
            label='Membrane Potential')
    ax1.axhline(V_thresh * 1000, color='#DC2626', linestyle='--',
                linewidth=1.2, label='Threshold (-55 mV)')
    ax1.axhline(V_rest * 1000, color='#16A34A', linestyle='--',
                linewidth=1.2, label='Resting Potential (-70 mV)')
    ax1.fill_between([40, 120], [-90, -90], [55, 55],
                    alpha=0.07, color='orange', label='Stimulus period')
    
    # 단계 레이블 (첫 번째 스파이크 기준)
    if spikes:
        sp = spikes[0]
        tx = t[sp] * 1000
        ax1.annotate('Depol.', xy=(tx + 0.3, 42), xytext=(tx + 4, 47),
                    fontsize=8, color='#1D4ED8',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax1.annotate('Repol.', xy=(tx + 0.8, 4), xytext=(tx + 5, 15),
                    fontsize=8, color='#7C3AED',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax1.annotate('Hyperpol.', xy=(tx + 2.2, -76), xytext=(tx + 6, -82),
                    fontsize=8, color='#DC2626',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    
    ax1.set_xlim(0, 150)
    ax1.set_ylim(-90, 60)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=11)
    ax1.set_title('Neuron Action Potential Simulation — LIF Model (Realistic Spike Shape)',
                fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.25)
    
    ax2.plot(t * 1000, I * 1e9, color='#D97706', linewidth=2)
    ax2.fill_between(t * 1000, 0, I * 1e9, alpha=0.25, color='#D97706')
    ax2.set_xlim(0, 150)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Input Current (nA)', fontsize=11)
    ax2.set_title('Injected Stimulus Current', fontsize=11)
    ax2.grid(True, alpha=0.25)
        
    plt.tight_layout(pad=1.5)
    plt.savefig('./result/sim1_action_potential.png', dpi=160,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("그래프 저장 완료")