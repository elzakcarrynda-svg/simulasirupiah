"""
Dashboard simulasi kurs USD/IDR (Streamlit)
Jalankan:  py -m streamlit run app.py
Paket   :  py -m pip install streamlit yfinance pandas numpy matplotlib
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

CSV = "kurs_usdidr.csv"
st.set_page_config(page_title="Dashboard Kurs USD/IDR", layout="wide")

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
st.sidebar.header("Data kurs")
if "seri" not in st.session_state:
    st.session_state.seri = baca_csv(CSV) if os.path.exists(CSV) else None

if st.sidebar.button("Ambil kurs terbaru (Yahoo)"):
    try:
        with st.spinner("Mengambil data..."):
            s = ambil_yahoo()
        s.to_csv(CSV)
        st.session_state.seri = s
        st.sidebar.success(f"Berhasil: {len(s)} data, terakhir {s.index[-1].date()}")
    except Exception as e:
        st.sidebar.error(f"Gagal mengambil data: {e}")

up = st.sidebar.file_uploader("Atau unggah CSV (kolom tanggal + kurs)", type="csv")
if up is not None and st.session_state.get("up_name") != up.name:
    try:
        st.session_state.seri = baca_csv(up)
        st.session_state.up_name = up.name
    except Exception as e:
        st.sidebar.error(f"CSV tidak terbaca: {e}")

seri = st.session_state.seri
st.title("Dashboard simulasi kurs USD/IDR")
if seri is None or len(seri) < 30:
    st.info("Belum ada data yang cukup. Klik 'Ambil kurs terbaru' atau unggah CSV di sidebar.")
    st.stop()

win = st.sidebar.slider("Jendela estimasi volatilitas (hari)", 20, min(250, len(seri) - 2), min(120, len(seri) - 2))
sig_base = estimasi(seri, win)

tab1, tab2, tab3 = st.tabs(["Kurs dan data", "Simulasi skenario", "Akurasi prediksi"])

# ---------------------------------------------------------------- tab 1
with tab1:
    r = np.log(seri).diff().dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurs terakhir", f"Rp{seri.iloc[-1]:,.0f}", f"{r.iloc[-1] * 100:+.2f}%")
    c2.metric("Tanggal terakhir", str(seri.index[-1].date()))
    c3.metric("Jumlah data", f"{len(seri)}")
    c4.metric(f"σ tahunan ({win} hari)", f"{sig_base:.1%}")
    st.line_chart(seri)
    besar = (r.abs() > 0.015).sum()
    if besar:
        st.warning(f"{besar} perubahan harian di atas 1,5%. Cek apakah itu data rusak atau memang gejolak nyata.")
    st.caption("Sumber: Yahoo Finance (IDR=X) atau CSV unggahan. Angka bisa sedikit berbeda dari Google/JISDOR.")
    with st.expander("Lihat tabel data"):
        st.dataframe(seri.sort_index(ascending=False).to_frame(), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    nama = st.selectbox("Skenario", list(SKEN))
    p = SKEN[nama]
    c1, c2, c3, c4 = st.columns(4)
    S0 = c1.number_input("Kurs awal S0", value=float(seri.iloc[-1]), step=10.0, key="s0")
    mu = c2.number_input("Drift μ per tahun", value=p["mu"], step=0.01, format="%.3f", key=f"mu_{nama}")
    sigma = c3.number_input("Volatilitas σ per tahun", value=float(round(sig_base * p["k"], 4)),
                            step=0.005, format="%.4f", key=f"sg_{nama}_{sig_base:.4f}")
    T = c4.slider("Horizon (hari bursa)", 20, 504, 252)
    n = st.slider("Jumlah lintasan", 1000, 10000, 5000, step=1000)
    pct = st.multiselect("Ambang kenaikan kurs dari S0 (%)", [3, 5, 10, 15, 20], default=[5, 10, 15])
    if p["jump"]:
        st.caption(f"Skenario ini menambah lompatan acak: {p['jump'][0]:.0f} per tahun, rata-rata {p['jump'][1]:.0%}, sd {p['jump'][2]:.0%}.")

    P = simulasi(S0, mu, sigma, p["jump"], T, n)
    akhir = P[:, -1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Median akhir", f"Rp{np.median(akhir):,.0f}")
    m2.metric("Persentil 5%", f"Rp{np.percentile(akhir, 5):,.0f}")
    m3.metric("Persentil 95%", f"Rp{np.percentile(akhir, 95):,.0f}")

    hari = pd.bdate_range(seri.index[-1], periods=T + 1)
    p5, p25, p50, p75, p95 = np.percentile(P, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i in range(30):
        ax.plot(hari, P[i], color="tab:blue", alpha=0.10, lw=0.7)
    ax.fill_between(hari, p5, p95, color="tab:blue", alpha=0.18, label="Rentang 5-95%")
    ax.fill_between(hari, p25, p75, color="tab:blue", alpha=0.25, label="Rentang 25-75%")
    ax.plot(hari, p50, color="black", lw=1.8, label="Median")
    for q in pct:
        ax.axhline(S0 * (1 + q / 100), color="gray", ls="--", lw=0.8)
    ax.set_ylabel("USD/IDR (Rp)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    tab = pd.DataFrame({
        "Ambang (Rp)": [f"{S0 * (1 + q / 100):,.0f} (+{q}%)" for q in pct],
        "Peluang kurs akhir ≥ ambang": [f"{(akhir >= S0 * (1 + q / 100)).mean():.1%}" for q in pct],
        "Peluang pernah ≥ ambang": [f"{(P.max(axis=1) >= S0 * (1 + q / 100)).mean():.1%}" for q in pct],
    })
    st.dataframe(tab, hide_index=True, width="stretch")

    with st.expander("Bandingkan keempat skenario (parameter bawaan)"):
        baris = []
        for nm, sp in SKEN.items():
            Q = simulasi(S0, sp["mu"], sig_base * sp["k"], sp["jump"], T, 3000)
            a = Q[:, -1]
            baris.append({"Skenario": nm, "Median akhir": f"{np.median(a):,.0f}",
                          "P5": f"{np.percentile(a, 5):,.0f}", "P95": f"{np.percentile(a, 95):,.0f}"})
        st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
    st.caption("Skenario adalah asumsi, bukan ramalan. Drift dan pengali volatilitas dipilih sendiri.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Uji ke belakang: dari tiap tanggal, model hanya memakai data sebelum tanggal itu, lalu hasilnya dibandingkan dengan kurs yang benar-benar terjadi. "
             "GBM dibandingkan dengan GBM-KF (GBM dimodifikasi Kalman Filter, seperti paper Hanifah dan Maulana) dan prediksi naif.")
    c1, c2, c3 = st.columns(3)
    H = c1.selectbox("Horizon prediksi (hari)", [1, 5, 10, 20], index=1)
    w2 = c2.slider("Jendela estimasi (hari)", 20, max(21, min(120, len(seri) - H - 5)), min(60, max(20, len(seri) - H - 6)))
    mode = c3.radio("Drift GBM", ["Tanpa drift", "Drift historis"], horizontal=True)
    with st.expander("Parameter Kalman Filter (GBM-KF)"):
        k1, k2 = st.columns(2)
        ratio = k1.number_input("R: noise pengamatan (kali σ²)", 0.01, 5.0, 0.10, step=0.05)
        qm = k2.number_input("Q drift: simpangan (kali σ)", 0.001, 0.5, 0.05, step=0.01, format="%.3f")
        st.caption("Keadaan: ln(kurs) dan drift harian. Q dan R memakai skala σ dari data, bukan 10⁻⁵ tetap seperti di paper. "
                   "Nilai bawaan dipilih tanpa melihat hasil uji; jangan disetel sampai hasilnya bagus lalu dilaporkan sebagai akurasi.")
    if len(seri) - H - w2 < 10:
        st.warning("Data terlalu pendek untuk backtest dengan pengaturan ini. Perkecil jendela atau horizon.")
    else:
        bt = backtest(seri, H, w2, mode).join(backtest_kf(seri, H, w2, ratio, qm))
        mp = lambda d, c: (abs(d["Aktual"] - d[c]) / d["Aktual"]).mean() * 100
        cv = lambda d, lo, hi: ((d["Aktual"] >= d[lo]) & (d["Aktual"] <= d[hi])).mean() * 100
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("MAPE GBM", f"{mp(bt, 'Median'):.3f}%")
        a2.metric("MAPE GBM-KF", f"{mp(bt, 'KF_Median'):.3f}%")
        a3.metric("MAPE naif (kurs = hari ini)", f"{mp(bt, 'Naif'):.3f}%")
        a4.metric("Jumlah prediksi", f"{len(bt)}")
        b1, b2 = st.columns(2)
        b1.metric("Cakupan pita 5-95% GBM", f"{cv(bt, 'P5', 'P95'):.0f}%", "target ±90%", delta_color="off")
        b2.metric("Cakupan pita 5-95% GBM-KF", f"{cv(bt, 'KF_P5', 'KF_P95'):.0f}%", "target ±90%", delta_color="off")

        k = int(len(bt) * 0.7)
        ring = pd.DataFrame({
            "Bagian": [f"In-sample (70% awal, {k} prediksi)", f"Out-sample (30% akhir, {len(bt) - k} prediksi)"],
            "MAPE GBM": [f"{mp(bt.iloc[:k], 'Median'):.3f}%", f"{mp(bt.iloc[k:], 'Median'):.3f}%"],
            "MAPE GBM-KF": [f"{mp(bt.iloc[:k], 'KF_Median'):.3f}%", f"{mp(bt.iloc[k:], 'KF_Median'):.3f}%"],
            "MAPE naif": [f"{mp(bt.iloc[:k], 'Naif'):.3f}%", f"{mp(bt.iloc[k:], 'Naif'):.3f}%"],
        })
        st.dataframe(ring, hide_index=True, width="stretch")

        fig2, ax2 = plt.subplots(figsize=(11, 4.5))
        ax2.fill_between(bt.index, bt["P5"], bt["P95"], color="tab:blue", alpha=0.18, label="Pita 5-95% GBM")
        ax2.plot(bt.index, bt["Median"], color="tab:blue", lw=1.2, label="Median GBM")
        ax2.plot(bt.index, bt["KF_Median"], color="tab:orange", lw=1.2, label="Median GBM-KF")
        ax2.plot(bt.index, bt["Aktual"], color="black", lw=1.5, label="Kurs aktual")
        ax2.set_ylabel("USD/IDR (Rp)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left")
        st.pyplot(fig2)
        st.caption("MAPE kecil hampir pasti, karena kurs harian jarang bergerak jauh; yang bermakna adalah selisih antar model dan terhadap prediksi naif. "
                   "Dengan 'Tanpa drift', MAPE GBM sama dengan naif. Cakupan jauh di bawah 90% berarti volatilitas diremehkan; jauh di atas berarti pita terlalu lebar. "
                   "Selisih kecil antar model belum tentu bermakna; paper acuan juga hanya menemukan selisih kecil pada data pendek.")