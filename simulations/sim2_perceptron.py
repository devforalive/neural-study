import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'

def sim2_perceptron():
    # ─────────────────────────────────────────────
    # 퍼셉트론 클래스
    # ─────────────────────────────────────────────
    class Perceptron:
        def __init__(self, lr=0.1, epochs=30):
            self.lr     = lr
            self.epochs = epochs
            self.history = []

        def step(self, z):
            return 1 if z >= 0.5 else 0

        def fit(self, X, y):
            self.w = np.array([0.0, 0.0])
            self.b = 0.0
            for epoch in range(self.epochs):
                err = 0
                for xi, yi in zip(X, y):
                    pred = self.step(np.dot(self.w, xi) + self.b)
                    delta = self.lr * (yi - pred)
                    self.w += delta * xi
                    self.b += delta
                    err += abs(yi - pred)
                self.history.append(err)

        def predict(self, X):
            return [self.step(np.dot(self.w, xi) + self.b) for xi in X]

    # AND gate
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,0,1])

    p = Perceptron(lr=0.1, epochs=30)
    p.fit(X, y)

    print("=== AND Gate Perceptron ===")
    for xi, yi in zip(X, y):
        pred = p.step(np.dot(p.w, xi) + p.b)
        print(f"  Input {xi.astype(int)} -> Pred: {pred}, Actual: {yi}")
    print(f"  Weights: w1={p.w[0]:.3f}, w2={p.w[1]:.3f}, bias={p.b:.3f}")

    # ─────────────────────────────────────────────
    # 그림 구성
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # (A) 학습 오차 수렴
    ax1 = axes[0]
    ax1.plot(range(1, len(p.history)+1), p.history,
            color='#2563EB', linewidth=2, marker='o', markersize=5, label='Total error per epoch')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Error', fontsize=12)
    ax1.set_title('(A) Perceptron Learning — AND Gate', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.2, max(p.history)+0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # (B) 결정 경계
    ax2 = axes[1]
    colors_map = {0: '#EF4444', 1: '#2563EB'}
    markers_map = {0: 'o', 1: '^'}
    labels_map  = {0: 'Output = 0 (F)', 1: 'Output = 1 (T)'}
    for cls in [0, 1]:
        mask = y == cls
        ax2.scatter(X[mask,0], X[mask,1], c=colors_map[cls],
                    marker=markers_map[cls], s=250, zorder=5, label=labels_map[cls],
                    edgecolors='black', linewidths=0.8)

    # 결정 경계선
    if abs(p.w[1]) > 1e-6:
        x_line = np.linspace(-0.3, 1.3, 200)
        y_line = (0.5 - p.b - p.w[0]*x_line) / p.w[1]
        ax2.plot(x_line, y_line, 'g-', linewidth=2.2, label='Decision Boundary')

    ax2.set_xlim(-0.4, 1.4)
    ax2.set_ylim(-0.4, 1.4)
    ax2.set_xlabel('Input x₁', fontsize=12)
    ax2.set_ylabel('Input x₂', fontsize=12)
    ax2.set_title('(B) Decision Boundary After Training', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout(pad=2)
    plt.savefig('./result/fig2_perceptron.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig2 saved.")
