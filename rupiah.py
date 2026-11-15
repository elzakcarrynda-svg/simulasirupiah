import yfinance as yf

d = yf.download("IDR=X", period="1y", progress=False)
d.columns = d.columns.get_level_values(0)       # rapikan kolom bertingkat

n0 = len(d)
d = d[(d["Close"] >= d["Low"]) & (d["Close"] <= d["High"])]   # buang Close di luar rentang
d = d[d.index.dayofweek < 5]                                  # buang Sabtu/Minggu
print(f"Baris dibuang: {n0 - len(d)} dari {n0}")

out = d[["Close"]].rename(columns={"Close": "kurs"})
out.index.name = "Tanggal"
out.to_csv("kurs_usdidr.csv")
print(out.tail())
print("Sigma tahunan hasil estimasi:", out["kurs"].pipe(lambda s: (s.pipe(__import__("numpy").log).diff().std() * 252**0.5)))