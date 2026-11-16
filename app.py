"""
Dashboard simulasi kurs USD/IDR (Streamlit)
Jalankan:  py -m streamlit run app.py
Paket   :  py -m pip install streamlit yfinance pandas numpy matplotlib
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

CSV = "kurs_usdidr.csv"
BG, BIRU, JINGGA, INK, REDUP = "#11161D", "#6FA8DC", "#E39A4B", "#E6EAF0", "#9AA5B4"
st.set_page_config(page_title="Kurs USD/IDR", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.9rem !important; padding-bottom: 0.2rem; }
[data-testid="stMetricLabel"] p { font-size: 0.85rem; color: #9AA5B4; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; }
.stAppDeployButton, footer { display: none; }
[data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"font.size": 10, "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.edgecolor": "#3A4452", "axes.labelcolor": REDUP,
                     "xtick.color": REDUP, "ytick.color": REDUP, "text.color": INK,
                     "grid.color": "#3A4452"})


def rp(x):
    """Format rupiah gaya Indonesia: Rp18.140"""
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def gaya(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))

# ----------------------------------------------------------------- data
def bersihkan(d):
    """Buang akhir pekan, Close di luar Low-High, dan lonjakan tak wajar."""
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[d.index.dayofweek < 5]
    d = d[(d["Close"] >= d["Low"]) & (d["Close"] <= d["High"])]
    med = d["Close"].rolling(7, center=True, min_periods=3).median()
    d = d[(d["Close"] / med - 1).abs() < 0.015]
    s = d["Close"].rename("kurs").dropna()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index.name = "Tanggal"
    return s


def ambil_yahoo():
    import yfinance as yf
    d = yf.download("IDR=X", period="2y", progress=False)
    return bersihkan(d)


def baca_csv(sumber):
    df = pd.read_csv(sumber, sep=None, engine="python")
    tgl = df.columns[0]
    kand = [c for c in df.columns if str(c).lower() in ("kurs", "close", "price", "jisdor", "rate")]
    val = kand[0] if kand else df.select_dtypes("number").columns[-1]
    s = pd.Series(pd.to_numeric(df[val], errors="coerce").values,
                  index=pd.to_datetime(df[tgl], errors="coerce"), name="kurs")
    s = s[s.index.notna()].dropna().sort_index()
    s.index.name = "Tanggal"
    return s


def estimasi(seri, window):
    r = np.log(seri).diff().dropna().tail(window)
    return r.std() * np.sqrt(252)


# ------------------------------------------------------------ simulasi
SKEN = {  # mu = drift tahunan, k = pengali sigma dasar, jump = (lambda/thn, rata2, sd)
    "Baseline": dict(mu=0.02, k=1.0, jump=None),
    "Depresiasi moderat": dict(mu=0.08, k=1.2, jump=None),
    "Krisis": dict(mu=0.15, k=2.0, jump=(2.0, 0.03, 0.02)),
    "Intervensi BI": dict(mu=0.03, k=0.7, jump=None),
}


@st.cache_data(show_spinner=False)
def simulasi(S0, mu, sigma, jump, T, n, seed=2026):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal((n, T))
    inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    if jump is not None:
        lam, mj, sj = jump
        inc += rng.poisson(lam * dt, size=(n, T)) * rng.normal(mj, sj, size=(n, T))
    P = np.exp(np.log(S0) + np.cumsum(inc, axis=1))
    return np.hstack([np.full((n, 1), S0), P])


def backtest(seri, H, win, mode):
    """Prediksi H hari ke depan dari tiap tanggal, memakai data sebelum tanggal itu saja.
    Kuantil GBM dihitung analitik (sama dengan hasil Monte Carlo untuk lintasan tak hingga)."""
    s = seri.values
    r = np.diff(np.log(s))
    rows = []
    for t in range(win, len(s) - H):
        rw = r[t - win:t]
        sd = rw.std(ddof=1) * np.sqrt(H)
        m = rw.mean() * H if mode == "Drift historis" else 0.0
        med = s[t] * np.exp(m)
        rows.append((seri.index[t + H], s[t + H], s[t], med,
                     s[t] * np.exp(m - 1.645 * sd), s[t] * np.exp(m + 1.645 * sd)))
    return pd.DataFrame(rows, columns=["Tanggal", "Aktual", "Naif", "Median", "P5", "P95"]).set_index("Tanggal")


def backtest_kf(seri, H, win, ratio, qm):
    """GBM-KF: filter Kalman pada ln(kurs) dengan keadaan [ln S, drift harian].
    Filter berjalan searah waktu, jadi prediksi di tanggal t hanya memakai data sampai t.
    Q = diag(sigma^2, (qm*sigma)^2), R = ratio*sigma^2, sigma = std return harian (jendela win)."""
    logs = np.log(seri.values)
    n = len(logs)
    ret = pd.Series(np.diff(logs, prepend=np.nan))
    sig = ret.rolling(win, min_periods=10).std().bfill().values
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    x = np.array([logs[0], 0.0])
    P = np.diag([sig[0] ** 2, (qm * sig[0]) ** 2])
    est = []  # (x_filtered, P_filtered, sigma) per hari
    for k in range(n):
        if k > 0:
            Q = np.diag([sig[k] ** 2, (qm * sig[k]) ** 2])
            x = F @ x
            P = F @ P @ F.T + Q
            R = ratio * sig[k] ** 2
            S = P[0, 0] + R
            K = P[:, 0] / S
            x = x + K * (logs[k] - x[0])
            P = P - np.outer(K, P[0, :])
        est.append((x.copy(), P.copy(), sig[k]))
    rows = []
    for t in range(win, n - H):
        x, P, sg = est[t]
        Q = np.diag([sg ** 2, (qm * sg) ** 2])
        for _ in range(H):
            x = F @ x
            P = F @ P @ F.T + Q
        sd = np.sqrt(P[0, 0] + ratio * sg ** 2)
        rows.append((seri.index[t + H], np.exp(x[0]),
                     np.exp(x[0] - 1.645 * sd), np.exp(x[0] + 1.645 * sd)))
    return pd.DataFrame(rows, columns=["Tanggal", "KF_Median", "KF_P5", "KF_P95"]).set_index("Tanggal")


# ------------------------------------------------------------- sidebar
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]


def tgl(d):
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


st.sidebar.subheader("Data")
if "seri" not in st.session_state:
    st.session_state.seri = baca_csv(CSV) if os.path.exists(CSV) else None

if st.sidebar.button("Ambil data terbaru (Yahoo Finance)"):
    try:
        with st.spinner("Mengambil data..."):
            s = ambil_yahoo()
        s.to_csv(CSV)
        st.session_state.seri = s
        st.sidebar.success(f"{len(s)} data, terakhir {s.index[-1].date()}")
    except Exception as e:
        st.sidebar.error(f"Data gagal diambil: {e}")

up = st.sidebar.file_uploader("Atau unggah CSV (kolom tanggal dan kurs)", type="csv")
if up is not None and st.session_state.get("up_name") != up.name:
    try:
        st.session_state.seri = baca_csv(up)
        st.session_state.up_name = up.name
    except Exception as e:
        st.sidebar.error(f"CSV tidak bisa dibaca: {e}")

seri = st.session_state.seri
st.title("Kurs USD/IDR")
if seri is None or len(seri) < 30:
    st.info("Data belum cukup (minimal 30 baris). Ambil data dari Yahoo Finance atau unggah CSV lewat panel kiri.")
    st.stop()

st.sidebar.caption(f"{len(seri)} data, {tgl(seri.index[0])} sampai {tgl(seri.index[-1])}")
win = st.sidebar.slider("Jendela volatilitas (hari)", 20, min(250, len(seri) - 2), min(120, len(seri) - 2))
sig_base = estimasi(seri, win)

r = np.log(seri).diff().dropna()
arah = "naik" if r.iloc[-1] >= 0 else "turun"
st.markdown(f"**{rp(seri.iloc[-1])}** per {tgl(seri.index[-1])}, {arah} {abs(r.iloc[-1]) * 100:.2f}% dari hari sebelumnya. "
            f"Volatilitas tahunan dari {win} hari terakhir sekitar {sig_base:.1%}.")

tab1, tab2, tab3 = st.tabs(["Kurs", "Simulasi", "Uji model"])

# ---------------------------------------------------------------- tab 1
with tab1:
    figd, axd = plt.subplots(figsize=(11, 3.8))
    axd.plot(seri.index, seri.values, color=BIRU, lw=1.4)
    axd.annotate(rp(seri.iloc[-1]), (seri.index[-1], seri.iloc[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", fontsize=9, color=INK)
    axd.margins(x=0.07)
    gaya(axd)
    st.pyplot(figd)
    besar = (r.abs() > 0.015).sum()
    if besar:
        st.caption(f"{besar} hari dengan perubahan di atas 1,5%. Periksa apakah itu data rusak atau gejolak sungguhan.")
    st.caption("Sumber: Yahoo Finance (IDR=X) atau CSV unggahan. Angkanya bisa sedikit berbeda dari JISDOR.")
    with st.expander("Tabel data"):
        st.dataframe(seri.sort_index(ascending=False).to_frame(), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    kiri, kanan = st.columns([1, 3], gap="large")
    with kiri:
        nama = st.selectbox("Skenario", list(SKEN))
        p = SKEN[nama]
        S0 = st.number_input("Kurs awal", value=float(seri.iloc[-1]), step=10.0, key="s0")
        mu = st.number_input("Drift (per tahun)", value=p["mu"], step=0.01, format="%.3f", key=f"mu_{nama}")
        sigma = st.number_input("Volatilitas (per tahun)", value=float(round(sig_base * p["k"], 4)),
                                step=0.005, format="%.4f", key=f"sg_{nama}_{sig_base:.4f}")
        T = st.slider("Horizon (hari bursa)", 20, 504, 252)
        n = st.slider("Jumlah lintasan", 1000, 10000, 5000, step=1000)
        pct = st.multiselect("Ambang kenaikan (%)", [3, 5, 10, 15, 20], default=[5, 10, 15])
        if p["jump"]:
            st.caption(f"Ditambah lompatan acak: {p['jump'][0]:.0f} kali per tahun, rata-rata {p['jump'][1]:.0%}, simpangan {p['jump'][2]:.0%}.")

    with kanan:
        P = simulasi(S0, mu, sigma, p["jump"], T, n)
        akhir = P[:, -1]
        st.markdown(f"Dalam {T} hari bursa, 90% lintasan berakhir antara **{rp(np.percentile(akhir, 5))}** "
                    f"dan **{rp(np.percentile(akhir, 95))}**, dengan median {rp(np.median(akhir))}.")
        hari = pd.bdate_range(seri.index[-1], periods=T + 1)
        p5, p25, p50, p75, p95 = np.percentile(P, [5, 25, 50, 75, 95], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i in range(20):
            ax.plot(hari, P[i], color=BIRU, alpha=0.18, lw=0.6)
        ax.fill_between(hari, p5, p95, color=BIRU, alpha=0.20, lw=0, label="5-95%")
        ax.fill_between(hari, p25, p75, color=BIRU, alpha=0.30, lw=0, label="25-75%")
        ax.plot(hari, p50, color=INK, lw=1.6, label="Median")
        for q in pct:
            y = S0 * (1 + q / 100)
            ax.axhline(y, color="#6B7685", ls=(0, (4, 3)), lw=0.8)
            ax.text(hari[0], y, f" +{q}%", va="bottom", fontsize=8.5, color=REDUP)
        gaya(ax)
        ax.legend(loc="upper left", frameon=False, ncol=3)
        st.pyplot(fig)
        if pct:
            tab = pd.DataFrame({
                "Ambang": [f"{rp(S0 * (1 + q / 100))} (+{q}%)" for q in pct],
                "Di atas ambang pada akhir periode": [f"{(akhir >= S0 * (1 + q / 100)).mean():.1%}" for q in pct],
                "Pernah menyentuh ambang": [f"{(P.max(axis=1) >= S0 * (1 + q / 100)).mean():.1%}" for q in pct],
            })
            st.dataframe(tab, hide_index=True, width="stretch")
        st.caption("Skenario adalah asumsi, bukan ramalan. Drift dan pengali volatilitas ditentukan sendiri.")

    with st.expander("Bandingkan semua skenario"):
        baris = []
        for nm, sp in SKEN.items():
            Q = simulasi(S0, sp["mu"], sig_base * sp["k"], sp["jump"], T, 3000)
            a = Q[:, -1]
            baris.append({"Skenario": nm, "Median akhir": rp(np.median(a)),
                          "Batas bawah": rp(np.percentile(a, 5)), "Batas atas": rp(np.percentile(a, 95))})
        st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.caption("Dari tiap tanggal, model hanya memakai data sebelum tanggal itu, lalu prediksinya dibandingkan dengan kurs yang benar-benar terjadi.")
    c1, c2, c3 = st.columns(3)
    H = c1.selectbox("Horizon prediksi (hari)", [1, 5, 10, 20], index=1)
    w2 = c2.slider("Jendela volatilitas (hari)", 20, max(21, min(120, len(seri) - H - 5)), min(60, max(20, len(seri) - H - 6)))
    mode = c3.radio("Drift GBM", ["Tanpa drift", "Drift historis"], horizontal=True)
    with st.expander("Parameter Kalman Filter (GBM-KF)"):
        k1, k2 = st.columns(2)
        ratio = k1.number_input("Noise pengamatan (kali σ²)", 0.01, 5.0, 0.10, step=0.05)
        qm = k2.number_input("Simpangan drift (kali σ)", 0.001, 0.5, 0.05, step=0.01, format="%.3f")
        st.caption("Keadaan filter: ln(kurs) dan drift harian. Nilai bawaan dipilih sebelum melihat hasil; "
                   "jangan disetel sampai hasilnya bagus lalu dilaporkan sebagai akurasi.")
    if len(seri) - H - w2 < 10:
        st.warning("Data terlalu pendek untuk pengaturan ini. Perkecil jendela atau horizon.")
    else:
        bt = backtest(seri, H, w2, mode).join(backtest_kf(seri, H, w2, ratio, qm))
        mp = lambda d, c: (abs(d["Aktual"] - d[c]) / d["Aktual"]).mean() * 100
        cv = lambda d, lo, hi: ((d["Aktual"] >= d[lo]) & (d["Aktual"] <= d[hi])).mean() * 100
        k = int(len(bt) * 0.7)
        kol = {"GBM": "Median", "GBM-KF": "KF_Median", "Naif": "Naif"}
        cakup = {"GBM": f"{cv(bt, 'P5', 'P95'):.0f}%", "GBM-KF": f"{cv(bt, 'KF_P5', 'KF_P95'):.0f}%", "Naif": "-"}
        hasil = pd.DataFrame([{
            "Model": nm,
            "MAPE seluruhnya": f"{mp(bt, c):.3f}%",
            "MAPE 70% awal": f"{mp(bt.iloc[:k], c):.3f}%",
            "MAPE 30% akhir": f"{mp(bt.iloc[k:], c):.3f}%",
            "Cakupan pita 90%": cakup[nm],
        } for nm, c in kol.items()])

        out = {nm: mp(bt.iloc[k:], c) for nm, c in kol.items()}
        best = min(out, key=out.get)
        if best == "Naif":
            st.markdown(f"Pada 30% data terakhir, tidak ada model yang mengalahkan prediksi naif (MAPE {out['Naif']:.3f}%).")
        else:
            sel = (out["Naif"] - out[best]) / out["Naif"] * 100
            st.markdown(f"Pada 30% data terakhir, MAPE terendah dimiliki **{best}** ({out[best]:.3f}%), "
                        f"{sel:.1f}% lebih baik daripada prediksi naif." + (" Selisih sekecil ini belum tentu bermakna." if sel < 5 else ""))
        st.dataframe(hasil, hide_index=True, width="stretch")
        st.caption(f"{len(bt)} prediksi diuji. Cakupan pita idealnya mendekati 90%.")

        fig2, ax2 = plt.subplots(figsize=(11, 4.2))
        ax2.fill_between(bt.index, bt["P5"], bt["P95"], color=BIRU, alpha=0.20, lw=0, label="Pita 5-95% GBM")
        ax2.plot(bt.index, bt["Median"], color=BIRU, lw=1.1, label="GBM")
        ax2.plot(bt.index, bt["KF_Median"], color=JINGGA, lw=1.1, label="GBM-KF")
        ax2.plot(bt.index, bt["Aktual"], color=INK, lw=1.4, label="Aktual")
        gaya(ax2)
        ax2.legend(loc="upper left", frameon=False, ncol=4)
        st.pyplot(fig2)
        with st.expander("Cara membaca hasil"):
            st.markdown(
                "- MAPE hampir pasti kecil karena kurs harian jarang bergerak jauh. Yang bermakna adalah selisih antar model dan terhadap prediksi naif.\n"
                "- Tanpa drift, MAPE GBM sama dengan naif.\n"
                "- Cakupan jauh di bawah 90% berarti volatilitas diremehkan; jauh di atas berarti pita terlalu lebar.\n"
                "- Selisih kecil antar model belum tentu bermakna. Paper acuan juga hanya menemukan selisih kecil pada data pendek.")