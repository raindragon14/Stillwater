# Evaluasi dan catatan penyelesaian manuskrip

Dokumen ini mencatat audit atas draf `docs/JOURNAL.md` pada commit awal
`25fb5c9` dan perubahan yang dilakukan untuk menghasilkan versi bersih. Draf
awal tetap dapat dilihat dengan:

```bash
git show 25fb5c9:docs/JOURNAL.md
```

## Ringkasan eksekutif

Draf awal adalah manuskrip metode yang kuat dan tertulis baik, tetapi belum
layak dikirim. Masalahnya bukan sekadar angka yang kosong: hipotesis utama
(luas Lévy menurunkan turnover) bertentangan dengan hasil akhir, teorema
diterapkan pada tanggal yang berada di luar domainnya, dan uji Rank IC memakai
koreksi sampel yang salah. Versi bersih menyelesaikan semua temuan yang dapat
diselesaikan dari artefak pipeline di
[`raindragon14/Stillwar`](https://github.com/raindragon14/Stillwar) rilis
`v1.0.0` (git SHA `870f93a`). Tidak ada angka yang dikarang; setiap angka
diambil dari artefak rilis atau dari manuskrip lengkap yang dihasilkan
pipeline yang sama.

## Metode audit

1. Membaca seluruh `docs/JOURNAL.md` dan menghitung token editorial.
2. Mencocokkan setiap klaim empiris dengan artefak:
   `table1.json`, `table2.json`, `table3.json`, `phase4.json`,
   `kkt_log.json`, `trials.json`, `gate_weights.json`, dan
   `forecasts_lstm.npz`.
3. Mencocokkan metode dengan kode dan konfigurasi:
   `configs/base.yaml`, `src/stillwar/models.py`, `portfolio.py`,
   `features.py`, `simulate.py`, `universe.py`, dan `data.py`.
4. Membandingkan dengan manuskrip lengkap `Stillwar/docs/JOURNAL.md` untuk
   detail yang tidak tersimpan di JSON (misalnya jumlah tanggal efektif dan
   persentase hari suspensi).
5. Menghitung ulang volatilitas pemicu dari panel harga dan `kkt_log.json`
   untuk mengganti angka volatilitas yang tidak memiliki definisi eksplisit.
6. Menjalankan `tools/check_manuscript.py` pada versi bersih.

## Temuan

| # | Tingkat | Temuan | Bukti | Penyelesaian |
|:--|:--|:--|:--|:--|
| 1 | Blocker | Seluruh hasil empiris masih placeholder: 49 `[RESULT]`, 8 `[CONFIRM]`, 18 cabang `[IF]`, dan 196 sel tabel kosong. | Draf awal. | Diisi dari artefak rilis; tabel 1–3 lengkap; tidak ada placeholder tersisa. |
| 2 | Blocker | Judul dan abstrak mengklaim luas Lévy **menurunkan** turnover. Hasil akhir menunjukkan sebaliknya: LogSig 23,08% vs LogSig-Inc 19,94% (Panel B), MDD 35,61% vs 31,89%. | `table1.json`; `docs/JOURNAL.md` §3.4. | Judul diubah menjadi “Lévy-Area Signal Activation”; abstrak, kontribusi (iii), §3.4, dan §4.1 ditulis ulang; hipotesis dinyatakan ditolak. |
| 3 | Blocker | Teorema 1 diterapkan pada seluruh 128 tanggal, padahal 75 tanggal memiliki $w^{-} > w_{\max}$ sehingga $w^{-}$ tidak fisibel dan teorema tidak berlaku. Draf juga mengatakan “on the rest the optimizer holds”, padahal 75 tanggal itu justru dipaksa bertransaksi. | `table3.json`: `cap_drift_traded = 75`, `hold_held = 20`; `phase4.json`. | Uji dinyatakan 100% cocok pada 53 tanggal fisibel interior; 75 tanggal dicatat sebagai cap-drift forced trades; Tabel 3 diberi baris cap-drift; §2.6, §2.7, §3.3, dan §4.2 dikualifikasi. |
| 4 | Mayor | Uji-t Rank IC memakai $n/5$ untuk deret mingguan. Target lima hari pada tanggal rebalance mingguan **tidak** tumpang tindih; koreksi $n/5$ hanya berlaku untuk deret harian ($n = 638$). | `stats.py::rank_ic_test`; `configs/base.yaml` (`effective_ratio_weekly: 1.0`). | §2.10 ditulis ulang: $n = 94$ (Panel A) dan 128 (Panel B), $df = n - 1$; koreksi $n/5$ hanya untuk deret harian. Nilai $p$ diambil dari artefak: ensemble mingguan $p = 0{,}0331$; harian $p = 0{,}0503$. |
| 5 | Mayor | $M = 1{,}235$ memasukkan sepuluh half-life dari eksperimen smoothing yang tidak ada di pipeline final. Artefak memakai $M = 1{,}225$ (1.200 grid + 20 non-inclusive + 5 klasik). | `trials.json`: `n_trials = 1225`; `trials.py`. | $M = 1{,}225$; eksperimen half-life dihapus; nilai Deflated Sharpe cocok dengan artefak. |
| 6 | Mayor | $\psi_{\kappa}$ didefinisikan atas seluruh tanggal rebalance, padahal tanggal cap-drift tidak memiliki $\Delta g_t$ berhingga. | `portfolio.py`: `valid_kkt` mengecualikan `delta_g = NaN`. | $\psi_{\kappa}$ didefinisikan atas $\mathcal{R}_{\mathrm{int}}$; $\psi_{5} = 75{,}50\%$ (rata-rata seed) dan $\psi_{5}^{0} = 0{,}00\%$. |
| 7 | Mayor | Cabang `[IF Y < 0]` akan mengklaim penolakan transaksi sebagai keputusan yang konsisten biaya, tanpa menyebut ketidakpastian. Estimasi akhir $-31{,}39$ bps/minggu tetapi $t = -0{,}84$, $p = 0{,}413$. | `phase4.json`. | Dilaporkan sebagai *economically suggestive, not statistically established*; $n = 20$ minggu disebut sebagai batas daya uji. |
| 8 | Mayor | Permintaan “penalty share vs feature share” tidak terdefinisi dengan baik: fitur tidak menurunkan turnover relatif terhadap Hist-MV berpenalti (20,36%/23,08% vs 1,98%/2,01%), sehingga “feature share” negatif. | `table1.json`. | Diganti dengan dekomposisi kualitatif: penalti menjelaskan penurunan turnover; fitur memulihkan Rank IC dan kinerja bersih relatif terhadap Hist-MV berpenalti. |
| 9 | Mayor | Alpha vs GMV dan “strategy-level $M = 7$” tidak ada di artefak. | `table1.json` hanya memuat `nw_alpha` vs EW; DSR memakai $M = 1{,}225$. | Klaim vs GMV dan $M = 7$ dihapus; Tabel 1 hanya melaporkan alpha vs EW. |
| 10 | Mayor | Keterangan Gambar 2 menyebut “heatmap”, padahal gambar aktual adalah dekomposisi batang sektor/konstituen. | `fig3_trigger_heatmap.png`. | Keterangan dan nama berkas diubah menjadi `fig2_trigger_decomposition.*`. |
| 11 | Mayor | DOCX memakai template “Journal of Education Research” (ISSN 2746-0738) dengan placeholder `x`, `xxxx`, dan “Judul Artikel Font Lato Size 10”; bidang jurnal tidak cocok dengan paper keuangan. | `word/header3.xml`, `word/footer1.xml`. | Markdown dibuat netral jurnal; DOCX diabaikan Git; jurnal target perlu dikonfirmasi. |
| 12 | Minor | Tanggal tidak konsisten: “2 January 2024” vs tanggal efektif 10 Januari 2024; “2 October 2026” vs observasi terakhir 1 Oktober 2026. | `features.py`; panel data. | Tanggal efektif dan jendela holding dikoreksi. |
| 13 | Minor | “40 saham LQ45” tanpa menjelaskan seleksi dari 53 kandidat. | `universe.py`. | Ditambahkan 53 kandidat, 2 tanpa snapshot, 51 efektif, 40 terpilih. |
| 14 | Minor | Persamaan (1) ditulis sebagai integral, sedangkan implementasi memakai jumlah hingga. | `features.py`; `MATH_SPEC.md`. | Diganti dengan bentuk jumlah hingga yang eksak untuk jalur piecewise-linear. |
| 15 | Minor | “The remaining subsections” dan “the four classical strategies and the unpenalized GMV five”. | Draf awal. | Diperbaiki menjadi “The following subsections” dan “the four classical strategies and the unpenalized GMV add five”. |
| 16 | Minor | Referensi [4] belum memuat judul perubahan PP 41/1994; [20] masih memiliki catatan `[verify]`; [36] belum ada. | Daftar pustaka. | [4] diperbaiki; catatan editorial dihapus; [36] ditambahkan; 36 referensi lengkap dan semuanya dikutip. Versi jurnal [20] belum diverifikasi (lihat daftar tindakan). |
| 17 | Minor | Tidak ada catatan tabel untuk unit turnover, rata-rata seed, $M$, dan denominator $\psi$. | Tabel 1–2. | Catatan tabel ditambahkan. |
| 18 | Minor | Alamat surel masih `[email address]`. | Draf awal. | Diisi `mrpandanarang@gmail.com` dari metadata proyek; konfirmasi penulis tetap diperlukan. |
| 19 | Minor | Draf memuat dua penulis; manuskrip lengkap Stillwar memuat satu penulis. | Draf vs `Stillwar/docs/JOURNAL.md`. | Dua penulis dipertahankan sesuai draf; perbedaan ini perlu dikonfirmasi. |

## Perubahan yang dilakukan

- **Judul**: “Lévy-Area Turnover Damping” diganti menjadi “Lévy-Area Signal
  Activation” karena hasil akhir menolak hipotesis damping.
- **Abstrak (Indonesia dan Inggris)**: diisi dengan angka final, kualifikasi
  “fisibel interior”, signifikansi dua sisi, dan temuan bahwa luas Lévy
  meningkatkan turnover.
- **§2.1**: ditambahkan pengaturan `auto_adjust=False`, penyesuaian
  `Adj Close / Close`, kebijakan forward-fill/volume nol, kandidat 53/51/40,
  dan tanggal efektif setelah purge/embargo.
- **§2.3**: persamaan Lévy area diubah ke bentuk jumlah hingga; ditambahkan
  catatan suspensi.
- **§2.4**: detail mini-batch 32 hari × 40 saham, pemilihan learning rate,
  dan metrik early stopping.
- **§2.5–§2.7**: ditambahkan fallback solver, short-circuit, toleransi
  eksekusi $10^{-4}$, batas boundary $10^{-6}$, dan definisi $\psi_{\kappa}$
  atas tanggal fisibel interior.
- **§2.9**: protokol hold (tidak ada order, biaya nol), tanggal efektif,
  jendela 470 hari untuk Panel A, dan unit turnover mingguan.
- **§2.10**: koreksi $n$ vs $n/5$, $M = 1{,}225$, dan cakupan DSR.
- **§3.1–§3.7**: seluruh tabel dan narasi diisi dari artefak; §3.3
  membedakan 53 tanggal interior dari 75 tanggal cap-drift; §3.4 menolak
  hipotesis damping; §3.5 memakai dekomposisi batang; §3.6 memakai
  dekomposisi kualitatif; §3.7 memakai Tabel 2 final.
- **§4.1–§4.3**: kesimpulan, batasan (termasuk cap-drift dan daya uji 20
  minggu), dan rekomendasi diselaraskan dengan hasil.
- **Lampiran A**: notasi $g = -\tilde{\mu} + \gamma\Sigma w^{-}$ diseragamkan;
  Tabel 3 ditambah baris cap-drift.
- **Gambar**: `fig1_kkt_spread.*` dan `fig2_trigger_decomposition.*`
  disalin dari artefak dan dirujuk dari Markdown.

## Yang masih memerlukan keputusan penulis

1. **Jurnal target dan template.** DOCX lama ditujukan untuk *Journal of
   Education Research*, yang tidak sesuai dengan paper keuangan. Markdown
   sekarang netral jurnal; konfirmasi jurnal target sebelum membuat DOCX baru.
2. **Daftar penulis, afiliasi, dan surel korespondensi.** Draf memuat dua
   penulis; manuskrip lengkap Stillwar memuat satu. Konfirmasi urutan,
   afiliasi, dan surel.
3. **Deklarasi.** Kontribusi penulis (CRediT), pendanaan, konflik kepentingan,
   ketersediaan data/kode, dan pengungkapan penggunaan AI (jika disyaratkan
   jurnal) belum ada.
4. **Gambar arsitektur dan ekstensi struktural.** Stillwar memiliki Figure 1
   (arsitektur) dan Figure 4/Table S1 (cap-drift buffer, kurva shrinkage,
   penalti berskala volatilitas). Versi bersih hanya memuat dua gambar dan
   tiga tabel; ekstensi tersebut saat ini disebut sebagai rekomendasi, bukan
   hasil. Putuskan apakah akan dimasukkan.
5. **Alpha vs GMV dan $p$-value per seed.** Keduanya tidak tersedia di
   artefak; saat ini dihilangkan/dikualifikasi. Jika diinginkan, perlu
   dihitung dari return series dan disimpan sebagai artefak baru.
6. **URL repositori.** Paper menunjuk ke Stillwar untuk kode/data. Jika
   repositori manuskrip ini dipublikasikan, perbarui URL dan rilis datanya.
7. **DOCX.** `docs/JOURNAL.docx` lama sudah usang, diabaikan Git, dan tidak
   boleh dipakai sebagai sumber. Regenerasi dari Markdown dengan template
   jurnal yang benar.
8. **Referensi [20].** Masih berupa SSRN working paper 2019. Catatan
   `[verify]` sudah dihapus, tetapi versi jurnal (jika ada) belum
   terverifikasi karena akses pencarian tidak tersedia saat penyelesaian ini.
   Periksa sebelum submit.

## Verifikasi

```bash
git log --oneline
git check-ignore -v docs/JOURNAL.docx
python tools/check_manuscript.py
```

Hasil yang diharapkan: dua commit (draf awal dan versi bersih), DOCX
terabaikan, dan pemeriksaan manuskrip lulus tanpa placeholder, tabel
persegi, referensi lengkap, serta gambar tersedia.

## Catatan tentang angka

Angka utama yang dipakai di manuskrip:

| Besaran | Nilai | Sumber |
|:--|:--|:--|
| Tanggal rebalance Panel B | 128 | `features.py`; `table3.json` |
| Tanggal fisibel interior / cap-drift | 53 / 75 | `table3.json`; `phase4.json` |
| Kesesuaian Teorema 1 | 33 trade, 20 hold, 0 boundary | `table3.json` |
| $\psi_{5}$ / $\psi_{5}^{0}$ | 75,50% / 0,00% | `table2.json` |
| Declined trades | $-31{,}39$ bps/minggu; $-627{,}79$ bps kumulatif; $t = -0{,}84$; $p = 0{,}413$ | `phase4.json` |
| Rank IC ensemble LogSig | $+0{,}0332$; $p = 0{,}0331$ | `table1.json` |
| Turnover LogSig vs LogSig-Inc (Panel B) | 23,08% vs 19,94% | `table1.json` |
| Volatilitas pemicu vs non-pemicu | 55,9% vs 37,3% | panel harga; `kkt_log.json` |
| $M$ | 1.225 | `trials.json` |
| Gate weights | 0,342 / 0,355 / 0,303 | `gate_weights.json` |
