"""휘켈 탐구 보고서용 계산·그림 코드 (Google Colab 또는 로컬 Python에서 실행)
필요: numpy, matplotlib   (Colab에는 이미 설치되어 있음)
그림의 글자는 한글 폰트 문제를 피하려고 영어로 썼다. 한글 라벨을 쓰려면 파일 맨 아래 메모 참고.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. 인접행렬 만들기 ----------
def path(n):                      # 탄소 n개가 일렬로 이어진 사슬
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A

def ring(n):                      # 사슬의 양 끝을 이은 고리
    A = path(n)
    A[0, -1] = A[-1, 0] = 1
    return A

# ---------- 2. 고유값 → 에너지 준위 (E = alpha + x*beta, beta<0 이므로 x가 클수록 에너지가 낮다) ----------
def levels(A):
    return np.sort(np.linalg.eigvalsh(A))[::-1]      # x가 큰 것(=낮은 에너지)부터

def pi_energy(x, n_electrons):
    """x: 큰 값부터 정렬된 고유값. 준위마다 전자 2개씩 아래부터 채운다. 반환값은 beta의 계수."""
    e, left = 0.0, n_electrons
    for v in x:
        put = min(2, left)
        e += put * v
        left -= put
        if left == 0:
            break
    return e

# ---------- 3. 표 계산: 사슬 DE, 고리 DE (기준 = 이중결합이 떨어진 구조, 에너지 n*beta) ----------
def table():
    print(" n   chain DE   ring DE   ring-chain")
    for n in (4, 6, 8, 10, 12):
        de_chain = pi_energy(levels(path(n)), n) - n
        de_ring  = pi_energy(levels(ring(n)),  n) - n
        print(f"{n:2d}   {de_chain:8.3f}  {de_ring:8.3f}  {de_ring - de_chain:9.3f}")

# ---------- 4. 그림 A: 사슬 길이에 따른 에너지 준위 (보고서 그림 4) ----------
def fig_levels():
    ns = [2, 4, 6, 8, 10, 14, 20]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, n in enumerate(ns):
        x = levels(path(n))
        for i, v in enumerate(x):
            filled = i < n // 2                       # 전자 n개 → 아래 n/2개 준위가 채워짐
            ax.hlines(v, j - 0.3, j + 0.3, color="k" if filled else "0.7", lw=2)
    ax.set_xticks(range(len(ns)), [f"n={n}" for n in ns])
    ax.set_ylabel("x  (E = alpha + x*beta; up = lower energy)")
    ax.axhline(0, ls="--", c="0.6")
    ax.set_title("Pi energy levels of linear chains")
    fig.tight_layout(); fig.savefig("fig_levels.png", dpi=200)

# ---------- 5. 그림 B: HOMO-LUMO 간격 (보고서 그림 5) ----------
def fig_gap():
    ns = np.arange(2, 31, 2)
    exact = 4 * np.sin(np.pi / (2 * (ns + 1)))
    approx = 2 * np.pi / (ns + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ns, exact, "ko-", mfc="w", label="4 sin(pi/(2(n+1)))")
    ax.plot(ns, approx, "--", c="0.5", label="2 pi/(n+1)")
    ax.set_xlabel("number of carbons n"); ax.set_ylabel("HOMO-LUMO gap (|beta|)")
    ax.legend(); fig.tight_layout(); fig.savefig("fig_gap.png", dpi=200)

# ---------- 6. 그림 C: 프로스트 원 (보고서 그림 6) ----------
def fig_frost(n_list=(4, 6, 8)):
    fig, axes = plt.subplots(1, len(n_list), figsize=(4 * len(n_list), 4))
    for ax, n in zip(axes, n_list):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(2 * np.cos(t), 2 * np.sin(t), c="0.7")
        ang = -np.pi / 2 + 2 * np.pi * np.arange(n) / n   # 한 꼭짓점이 아래
        px, py = 2 * np.cos(ang), 2 * np.sin(ang)
        ax.plot(np.append(px, px[0]), np.append(py, py[0]), "k-")
        ax.plot(px, py, "ko")
        ax.axhline(0, ls="--", c="0.6")                   # 중심 높이 = alpha (x=0)
        ax.set_title(f"n = {n}"); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig("fig_frost.png", dpi=200)

# ---------- 7. 그림 D: 고리 DE - 사슬 DE (보고서 그림 7) ----------
def fig_de():
    ns = [4, 6, 8, 10, 12]
    diff = [pi_energy(levels(ring(n)), n) - pi_energy(levels(path(n)), n) for n in ns]
    colors = ["0.7" if n % 4 == 2 else "w" for n in ns]   # 4m+2 는 회색
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(n) for n in ns], diff, color=colors, edgecolor="k")
    ax.axhline(0, c="k"); ax.set_xlabel("n"); ax.set_ylabel("ring DE - chain DE (|beta|)")
    fig.tight_layout(); fig.savefig("fig_de.png", dpi=200)

if __name__ == "__main__":
    table(); fig_levels(); fig_gap(); fig_frost(); fig_de()
    print("saved: fig_levels.png fig_gap.png fig_frost.png fig_de.png")

# 한글 라벨을 쓰고 싶으면 (Colab):  !apt-get -qq install fonts-nanum   후
#   plt.rcParams["font.family"] = "NanumGothic";  plt.rcParams["axes.unicode_minus"] = False
