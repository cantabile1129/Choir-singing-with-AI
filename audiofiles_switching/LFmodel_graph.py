import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


# =========================
# パラメータ設定
# =========================
T0 = 1.0

Tp = 0.30
Te = 0.45
Tc = 0.82

E1 = 1.0

alpha = 3.0
wg = np.pi / Tp

# Tcで「ほぼ0」にするための設定
eps = 0.03   # Tcでの残差割合（小さいほど滑らか）
b = -np.log(eps) / (T0 - Tc)

N = 100
t = np.linspace(0, T0, N)


# =========================
# E2 を Te の連続条件から決定
# =========================
u_Te = E1 * np.exp(alpha * Te) * np.sin(wg * Te)

E2 = -u_Te / (1.0 - np.exp(-b * (T0 - Te)))


# =========================
# u'(t) の計算
# =========================
u_prime = np.zeros_like(t)

# 開放期
idx_open = (t >= 0) & (t <= Te)
u_prime[idx_open] = (
    E1 * np.exp(alpha * t[idx_open]) * np.sin(wg * t[idx_open])
)

# 閉鎖期
idx_close = (t > Te) & (t <= Tc)
u_prime[idx_close] = (
    -E2 * (
        np.exp(-b * (t[idx_close] - Te))
        - np.exp(-b * (T0 - Te))
    )
)

# =========================
# sampling points 軸
# =========================
x = np.arange(N)


# =========================
# 縦軸範囲
# =========================
ymin = np.min(u_prime) - 0.15
ymax = np.max(u_prime) + 0.15


# =========================
# プロット
# =========================
plt.figure(figsize=(8, 4.5))

plt.plot(x, u_prime, color="blue", linewidth=2)

ax = plt.gca()

# 横軸（amplitude=0）を黒太線
ax.axhline(0, color="black", linewidth=2)

# フェーズ境界
ax.axvline(Te * (N - 2), color="red", linewidth=2)
ax.axvline(Tc * (N - 2), color="red", linewidth=2)

# 軸設定
ax.set_xlim(0, N - 1)
ax.set_ylim(ymin, ymax)

ax.set_xlabel("Sampling points", fontsize=14)
ax.set_ylabel("Amplitude", fontsize=14)

# 縦軸目盛り：0 と −Ee
ax.set_yticks([0, -E2])
ax.set_yticklabels(["0", r"$-E_2$"])

ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()
