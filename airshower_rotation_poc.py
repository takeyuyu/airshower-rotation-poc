# -*- coding: utf-8 -*-
"""
エアシャワー「回転」判定の最小PoC ＜第2作・後編＞

エアシャワーの正式な入り方は、両腕を上げて、その場でゆっくり
360度回転し、全身を気流にさらすこと。
前編（第1作）では、このうち "腕を上げているか" を判定した。
後編（本PoC）は、残り半分の "ちゃんと回ったか" を判定する。

噴霧PoC・前編と同じ思想：
  対象の「形」を直接見るのではなく、捉えやすい1次元信号の
  「時間変化」を使う。
  噴霧では「明るさ」、前編では「腕の高さ」を使った。
  ここでは「体の向き（角度）」の時間変化を使う。
  肩の左右の幅から体の向きを推定し、累積回転角を追う。

確かめたいこと：
  回り方は、人によって揃わない。
  ゆっくり一周して全面に風を当てる人・速くクルッと回るだけの人・
  半分だけ回ってやめる人。
  この3つを、体の向きの時間波形だけで切り分けられるか。
  そして「速い回転（角度は360度に届くが、各向きの滞留が足りない）」を
  見抜けるか。

  ＜前編との対称＞
  前編：上げた回数が同じでも、高さと保持で別物だった。
  後編：回った角度が同じ（360度）でも、滞留時間で別物になる。

※ 実データは一切使っていない。すべて合成データ。
  原理が成立するかを確かめる第一歩であり、実機動作の保証ではない。
  特に回転は、背面を向くと骨格推定が乱れるため、実機ではここで
  扱うほど素直な信号は得にくい。その難しさは記事本文で述べる。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]:
    try:
        fm.fontManager.addfont(path)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

FPS = 30
DURATION = 6.0     # エアシャワー6秒間（この間にゆっくり1周するのが正式）
N = int(FPS * DURATION)
t = np.linspace(0, DURATION, N)


def _smooth(x, k=5):
    # 端を値で延長してから移動平均（端のアーティファクトを防ぐ）
    pad = k // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, np.ones(k) / k, mode="same")[pad:-pad]


def synth_orientation(kind):
    """
    体の向き（角度・度）の時間波形を合成する。
    0度=正面、90度=真横、180度=背面 …と連続的に増えていく想定。
    骨格推定で左右の肩の見かけの幅から向きを割り出した代理データ。

    正式手順は「6秒かけてゆっくり360度」。だから理想は、
    0度から360度へ、ほぼ一定の速さでなめらかに増える直線。
    """
    if kind == "ゆっくり一周":
        # 6秒かけて 0→360度を一定速度で。各向きに均等に時間をかける。
        ang = np.linspace(0, 360, N)
        ang += 4.0 * rng.standard_normal(N)

    elif kind == "速い回転":
        # 最初の1.5秒で一気に360度回り、あとは正面で止まっている。
        # 一周はするが、各向きの滞留が足りない（風が当たる時間が短い）。
        fast = int(0.25 * N)
        ang = np.zeros(N)
        ang[:fast] = np.linspace(0, 360, fast)
        ang[fast:] = 360
        ang += 4.0 * rng.standard_normal(N)

    elif kind == "半周でやめる":
        # 180度（背中まで）回ったところでやめてしまう。
        half = int(0.5 * N)
        ang = np.zeros(N)
        ang[:half] = np.linspace(0, 180, half)
        ang[half:] = 180
        ang += 4.0 * rng.standard_normal(N)

    else:
        raise ValueError(kind)

    # 角度は0〜360度の範囲に収める（到達角度が360を超えて表示されないように）
    return np.clip(_smooth(ang), 0, 360)


def features(ang):
    """
    向きの波形から3つの素朴な特徴量を出す。引き算と数えるだけ。
      - 到達角度（最後にどこまで回ったか）              = total
      - 向きの種類（正面/横/背面/逆横 の4区画を踏んだ数） = quad
      - 最小滞留（4区画それぞれに居た時間の、いちばん短いコマ割合）= dwell

    ※ ここが肝。"何度回ったか" だけ見ると、速くクルッと回った人も
       360度で合格になる。大事なのは "各向きで十分に時間を過ごしたか"。
       いちばん滞在が短い向き＝いちばん風が当たっていない向きを見る。
    """
    total = float(ang.max())

    # 4区画（0-90, 90-180, 180-270, 270-360度）のどこに何コマ居たか
    bins = np.floor((ang % 360) / 90).astype(int)
    bins = np.clip(bins, 0, 3)
    counts = np.array([np.sum(bins == q) for q in range(4)])

    quad = int(np.sum(counts > 0))            # 踏んだ区画の数（最大4）
    reached = total >= 350                    # ほぼ一周したか
    if reached:
        dwell = float(counts.min() / N)       # 最も滞在の短い向きの時間割合
    else:
        dwell = 0.0                            # 一周してなければ滞留は問わない

    return total, quad, dwell


def judge(total, quad, dwell):
    """
    切り分けロジック。到達角度だけでは「速い回転」を見抜けない、が肝。
    """
    if total < 200:
        return "未実施", f"{int(total)}度しか回っていない（一周していない）"
    if dwell < 0.10:
        return "形だけ", f"一周はしたが、向きが偏って滞留が足りない（最小滞留{dwell:.2f}）"
    return "良好", f"一周し、各向きに十分とどまっている（最小滞留{dwell:.2f}）"


kinds = ["ゆっくり一周", "速い回転", "半周でやめる"]
signals = {k: synth_orientation(k) for k in kinds}

print("=" * 62)
print(" エアシャワー 回転 判定PoC ＜第2作・後編＞（合成データ）")
print("=" * 62)
results = {}
for k in kinds:
    total, quad, dwell = features(signals[k])
    verdict, reason = judge(total, quad, dwell)
    results[k] = (total, quad, dwell, verdict, reason)
    print(f"\n■ 入力：{k}")
    print(f"   到達角度     : {total:.0f} 度")
    print(f"   踏んだ向き   : {quad} / 4 区画")
    print(f"   最小滞留     : {dwell:.2f}")
    print(f"   → 判定：【{verdict}】（{reason}）")
print("\n" + "=" * 62)
print(" 注目点：『ゆっくり一周』も『速い回転』も、到達角度は同じ360度。")
print("         角度だけ見ると、どちらも一周＝合格に見える。")
print("         各向きの滞留時間で見ると、別物だと分かる。")
print("=" * 62)

# ---- 作図1：3つの向きの時間波形 ----
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
colors = {"ゆっくり一周": "#2E7D32", "速い回転": "#F9A825", "半周でやめる": "#C62828"}
for ax, k in zip(axes, kinds):
    ax.plot(t, signals[k], color=colors[k], lw=1.8)
    ax.axhline(360, color="gray", ls=":", lw=1)   # 一周の目安
    ax.set_ylim(-20, 400)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel("体の向き（度）")
    v = results[k][3]
    ax.set_title(f"{k}  →  判定【{v}】", loc="left", fontsize=12, color=colors[k])
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("時間（秒）")
fig.suptitle("エアシャワー内の体の向き：3人ぶんの時間波形", fontsize=13)
fig.tight_layout()
fig.savefig("rotation_waves.png", dpi=130)

# ---- 作図2：角度では分けられない／滞留で分かれる ----
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

# 左：到達角度だけ見ると「ゆっくり」と「速い」が同じ360度で並ぶ
axL.bar([k for k in kinds], [results[k][0] for k in kinds],
        color=[colors[k] for k in kinds])
axL.axhline(360, color="gray", ls=":", lw=1)
axL.set_title("到達『角度』だけ見ると…", fontsize=12)
axL.set_ylabel("到達角度（度）")
axL.set_ylim(0, 430)
axL.text(0.03, 0.97, "「速い回転」も到達角度は360度。\n角度だけ見ると『ゆっくり』と\n見分けがつかない",
         transform=axL.transAxes, ha="left", va="top", fontsize=9.5,
         bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#F9A825"))

# 右：到達角度×最小滞留の平面に置くと、くっきり分かれる
for k in kinds:
    total, quad, dwell, v, _ = results[k]
    axR.scatter(total, dwell, s=260, color=colors[k], edgecolor="black",
                zorder=3, label=f"{k}（{v}）")
    axR.annotate(k, (total, dwell), textcoords="offset points",
                 xytext=(8, 8), fontsize=10)
axR.axvline(350, color="gray", ls="--", lw=1)
axR.axhline(0.10, color="gray", ls="--", lw=1)
axR.set_xlabel("到達角度（度）")
axR.set_ylabel("最小滞留（最も風が当たらない向きの時間）")
axR.set_title("『角度』と『滞留』で見ると、くっきり分かれる", fontsize=12)
axR.set_xlim(0, 420)
axR.set_ylim(-0.02, 0.35)
axR.grid(alpha=0.25)
axR.legend(loc="upper center", fontsize=9)

fig2.suptitle("一周しても、中身は違う", fontsize=13)
fig2.tight_layout()
fig2.savefig("rotation_separation.png", dpi=130)

print("\n図を出力：rotation_waves.png / rotation_separation.png")
