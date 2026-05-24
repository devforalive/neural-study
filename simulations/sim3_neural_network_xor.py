import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sim3_neural_network_xor():
    # ── 활성화 함수 ──
    def sigmoid(z):  return 1 / (1 + np.exp(-z))
    def sig_d(z):    return sigmoid(z) * (1 - sigmoid(z))

    # ── 신경망 초기화 ──
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5   # 입력(2) → 은닉(4)
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 1) * 0.5   # 은닉(4) → 출력(1)
    b2 = np.zeros(1)

    # ── XOR 데이터 ──
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]])

    # ── 학습 ──
    lr = 0.5
    loss_history = []

    for epoch in range(5000):
        # 순전파
        z1  = X @ W1 + b1
        h   = sigmoid(z1)
        z2  = h @ W2 + b2
        out = sigmoid(z2)

        # 손실 (MSE)
        loss = np.mean((out - y) ** 2)
        loss_history.append(loss)

        # 역전파
        dOut = 2 * (out - y) / len(y)
        dz2  = dOut * sig_d(z2)
        W2  -= lr * (h.T @ dz2)
        b2  -= lr * dz2.sum(axis=0)

        dh  = dz2 @ W2.T
        dz1 = dh * sig_d(z1)
        W1 -= lr * (X.T @ dz1)
        b1 -= lr * dz1.sum(axis=0)

    # ── 결과 출력 ──
    print("=== 2층 신경망 XOR 학습 결과 ===")
    for xi, yi in zip(X, y):
        z1  = xi @ W1 + b1
        h   = sigmoid(z1)
        z2  = h @ W2 + b2
        pred = sigmoid(z2)[0]
        print(f"  입력 {xi.astype(int)} → 출력: {pred:.4f}  (정답: {int(yi[0])})")

    # ── 그래프 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # 손실 수렴 그래프
    axes[0].plot(loss_history, color='#2563EB', linewidth=1.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss — 2-Layer Neural Network (XOR)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 결정 경계 시각화
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z1g  = grid @ W1 + b1
    hg   = sigmoid(z1g)
    z2g  = hg @ W2 + b2
    outg = sigmoid(z2g).reshape(xx.shape)

    axes[1].contourf(xx, yy, outg, levels=50, cmap='RdBu_r', alpha=0.6)
    axes[1].contour(xx, yy, outg, levels=[0.5], colors='black', linewidths=2)

    colors = {0: '#EF4444', 1: '#2563EB'}
    markers = {0: 'o', 1: '^'}
    labels  = {0: 'XOR=0', 1: 'XOR=1'}
    for cls in [0, 1]:
        mask = y.ravel() == cls
        axes[1].scatter(X[mask, 0], X[mask, 1],
                        c=colors[cls], marker=markers[cls], s=300,
                        edgecolors='black', linewidths=1.2,
                        zorder=5, label=labels[cls])

    axes[1].set_xlim(-0.3, 1.3)
    axes[1].set_ylim(-0.3, 1.3)
    axes[1].set_xlabel('Input x1', fontsize=12)
    axes[1].set_ylabel('Input x2', fontsize=12)
    axes[1].set_title('Decision Boundary — XOR Problem', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('./result/sim3_xor_result.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print("그래프 저장 완료: sim3_xor_result.png")
