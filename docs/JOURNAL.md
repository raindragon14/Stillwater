# When the Optimizer Refuses to Trade: A KKT No-Trade Zone and Lévy-Area Turnover Damping under Weak Signals in LQ45 Equities

Muhammad Reihan Pandanarang¹,\*


(1) Mathematics Department, Universitas Negeri Semarang

\*Corresponding author

([email address])

Abstrak

Portofolio mean-variance yang diseimbangkan ulang berdasarkan prakiraan yang bising dapat kehilangan sebagian imbal hasilnya karena biaya transaksi. Penelitian ini mengkaji bagaimana sinyal imbal hasil yang lemah dan dipelajari dari data melintasi pita tanpa transaksi menjadi transaksi nyata pada 40 saham LQ45 di Bursa Efek Indonesia. Jaringan bergerbang multiskala memprakirakan imbal hasil mingguan dari log-signature orde dua jalur OHLCV harian, yang luas Lévy-nya merekam urutan gerak harga-volume, lalu prakiraan masuk ke masalah mean-variance long-only berbatas dengan penalti beli dan jual asimetris sesuai biaya BEI. Uji tanpa transaksi eksplisit dari kondisi optimalitas menyatakan bahwa portofolio saat ini optimal jika dan hanya jika selisih gradien antara aset yang dapat dijual dan dibeli tidak melebihi penalti pulang-pergi. Uji ini sesuai dengan solver pada [X]% tanggal penyeimbangan, dan transaksi yang ditolak akan [merugi/untung] [Y] basis poin setelah biaya. Rank IC sebesar [ ] (p = [ ]), dan luas Lévy menjelaskan [ ]% penurunan turnover di luar efek pemulusan.

**Kata Kunci:** kondisi KKT; zona tanpa transaksi; log-signature; biaya transaksi; LQ45

Abstract

Rebalancing on noisy forecasts can cost a mean-variance portfolio part of its return. We study how a weak, learned return signal crosses a no-trade band into actual trades for 40 LQ45 stocks on the Indonesia Stock Exchange. A multiscale gated network forecasts weekly returns from order-2 log-signatures of daily OHLCV paths, whose Lévy areas record the order of price-volume moves, and feeds a long-only, box-constrained mean-variance problem with asymmetric buy and sell penalties tied to IDX costs. An explicit no-trade test from the optimality conditions states that the current portfolio is optimal if and only if the gradient spread between sellable and buyable assets is at most the round-trip penalty. It agrees with the solver at [X]% of rebalance dates, and the declined trades would have [lost/gained] [Y] basis points after costs. Rank IC is [ ] (p = [ ]), and Lévy areas account for [ ]% of the turnover reduction beyond smoothing.

**Keyword:** KKT conditions; no-trade region; log-signature; transaction costs; LQ45

# 1. INTRODUCTION

Mean-variance optimization remains the reference framework for portfolio construction [1], yet it is notoriously fragile in practice. Because the optimizer treats estimated expected returns as if they were true, it systematically overweights assets whose returns are overestimated, a behavior Michaud termed error maximization [2]; the problem is severe enough that the naive equally weighted portfolio is difficult to beat out of sample [3]. This fragility becomes costly once trading is not free. Each revision of the forecasts can move the optimal weights, and every move incurs a cost that is certain, whereas the benefit of the revised forecast is not. On the Indonesia Stock Exchange (IDX), the certain part is substantial: a typical retail broker charges 0.15% on purchase and 0.25% on sale, a 0.40% round trip of which 0.1% is the final income tax on sales [4, 5]. Consequently, a portfolio that rebalances weekly on noisy forecasts can lose a material share of its return to trading alone.

Every such rebalancing decision passes through four ingredients in sequence: the market history that is observed, the beliefs about returns and risk estimated from it, the rule that maps these beliefs to portfolio weights, and the friction of moving from the current portfolio to the new one. Research on this problem can be organized into four lines by the ingredient each improves.

The first line changes how market history is represented. The signature of a path, a central object in rough path theory [6, 7], summarizes a multivariate time series through its iterated integrals. At order two, the log-signature reduces to the path increments and the Lévy areas: signed areas that record whether one coordinate moves before or after another, for example whether a surge in volume precedes or follows a price move [7]. Signature portfolios can uniformly approximate path-dependent portfolio functions [8], signature-based trading extends the mean-variance framework to path-dependent signals [9], and randomized signatures estimate drifts for portfolio selection [10].

The second line takes this representation as given and improves the beliefs estimated from it. Shrinkage estimators stabilize the covariance matrix [11], and machine-learning models extract return signals from large feature sets; even so, equity returns have a very low signal-to-noise ratio, and the best models achieve only modest predictive accuracy [12]. Worse, forecasting models trained on weakly predictable targets can collapse toward the cross-sectional mean, which motivates losses that combine calibration with cross-sectional ranking [13].

Because beliefs remain noisy however they are estimated, the third line makes the decision rule itself less sensitive to estimation error. Robust formulations optimize over uncertainty sets that correspond to confidence regions for the estimated parameters [14], norm constraints on the portfolio-weight vector help portfolios perform well out of sample [15], and decision-focused learning trains forecasting models on the error of the decisions they induce rather than on prediction error [16].

The fourth line acts on friction and controls trading directly. Penalizing turnover in the objective is equivalent to a form of regularization that reduces the impact of estimation error [17, 18]. With predictable returns and quadratic trading costs, the optimal policy trades only partially toward an aim portfolio that weights signals by their persistence [19], whereas proportional costs create a no-trade region, in which holding the current portfolio is optimal [20, 21].

These four lines have developed largely in parallel. In recent signature-based and machine-learning portfolio studies, for example, the decision of when to trade is delegated to a cost term or a chosen threshold rather than characterized [8, 10]. Such a characterization is within reach: whether holding the current portfolio is optimal can be read from standard subdifferential optimality conditions [22, 23]. Closely related no-trade characterizations exist for mean-variance rebalancing with transaction costs [20], multiperiod portfolios with general costs [21] and convex formulations and algorithms for transaction-cost portfolio problems [24, 25], and the two-threshold (violating-pair) optimality condition of sequential minimal optimization (SMO) is an analogue outside finance [26]. What is missing is an explicit, directly computable specialization to the fully invested, long-only, box-constrained case without a risk-free asset, used as a per-date diagnostic. Without such a diagnostic, it remains unclear how a weak, learned return signal crosses a no-trade band into actual trades: a forecast that is statistically indistinguishable from noise may still generate substantial turnover, or none at all. Answering it requires reading, at each rebalance date, whether the forecast or the risk model drives the decision to trade, and whether the trades that are declined would have paid for their costs.

The Indonesian literature has not yet reached this question: LQ45 portfolio studies form Markowitz allocations from historical returns [27, 28] or pass LSTM/BiLSTM price forecasts built on technical-indicator inputs to a Markowitz model without turnover or cost terms [29], while IDX forecasting studies apply deep-learning models to price data [30] and volume-based technical indicators to LQ45 returns [31]. To our knowledge, no IDX study encodes the joint, order-dependent price–volume path, for example through Lévy areas, or links such features to a cost-aware mean-variance allocation.

We address these gaps in a setting where the return signal is weak. A compact multiscale gated network forecasts weekly returns for 40 LQ45 constituents from order-2 log-signatures of daily OHLCV paths over 5-, 20- and 60-day windows and is trained with a loss that balances calibration and ranking. The forecasts feed a long-only, box-constrained mean-variance problem with a Ledoit–Wolf covariance [11] and asymmetric buy and sell turnover penalties tied to the IDX purchase and sale costs. Rather than asking whether the signal beats the market, we ask which components of this pipeline protect the investor from trading on noise, and we answer it through the optimality conditions of the problem itself.

Answering this question yields three contributions. (i) We state an explicit, multiplier-free no-trade test for the turnover-penalized problem under budget and box constraints: the current portfolio is optimal if and only if the largest gradient component among assets that can be sold exceeds the smallest among assets that can be bought by at most the round-trip cost $\eta_{b} + \eta_{s}$. The test follows from standard subdifferential optimality conditions [22, 23] and shares the structure of the violating-pair condition of SMO [26]; we use it as a per-date diagnostic and verify it against the solver at every rebalance date.

(ii) We introduce the signal-activation ratio $\psi_{\kappa}$, the share of rebalance dates on which the gradient spread crosses this threshold, and compare it with a constant-forecast benchmark; the gap measures how much of the trading the forecast itself drives and thereby links forecast collapse to realized trading.

(iii) We examine whether path-order features dampen the turnover induced by the trading signal, using an ablation that removes the Lévy areas, a decomposition of the constituents that trigger trades and a two-panel out-of-sample evaluation with deflated Sharpe ratios [32], in which all learned baselines share the same loss and penalty. Section 2 describes the methods, Section 3 presents and discusses the results, and Section 4 concludes.

# 2. METHODOLOGY

## 2.1 Data and sample

The sample consists of 40 constituents of the LQ45 index of the Indonesia Stock Exchange. Daily open, high, low and close prices and volume (OHLCV) are downloaded once from Yahoo Finance for 2 January 2017 to 2 October 2026 [CONFIRM: price-adjustment setting of the download, fixed in the configuration], archived as a versioned snapshot with a SHA-256 manifest and never patched: missing prices are carried forward only, never filled from later dates. The universe follows a rule computed from the data rather than a typed list. From [CONFIRM: candidate pool, i.e. the LQ45 constituent list and its date], we keep the 40 stocks with the largest median daily trading value (close × volume) over calendar year 2017 among those with a complete history to 2 October 2026. The ranking uses only information available before the first training date, whereas the completeness requirement conditions on survival, a limitation we return to in Section 4.2. The 2017 data serve only this ranking; features and training start in 2018.

Portfolios are rebalanced weekly, every five trading days. At the close of a decision date $t$ the model forecasts, for each stock $i$, the five-day log return $y_{t,i} = \log\left( P_{t + 6,i}/P_{t + 1,i} \right)$ earned by a position opened at the close of $t + 1$ and held over days $t + 2$ to $t + 6$, so the target never includes the execution day. The sample is split chronologically into a warm-up period (2 January to 27 March 2018) that fills the longest feature window, a training period (28 March 2018 to 30 June 2023), a validation period (3 July to 29 December 2023) and a test period (2 January 2024 to 30 December 2025). The last six days of the training and validation periods are purged, so that no label overlaps the next period, and an embargo of six trading days separates consecutive periods. Panel A is the test period. Panel B is the full out-of-sample period from 2 January 2024 to 2 October 2026, which adds the 2026 continuation; both panels come from one continuous run (Section 2.9).

## 2.2 Transaction-cost model

Exchange, clearing and settlement fees and the guarantee-fund contribution levied by IDX, KPEI and KSEI total 0.0433% of traded value per side including value-added tax [33]; sales also carry the 0.1% final income tax [4]. Online retail brokers quote about 0.15% on purchases and 0.25% on sales [5, 34], and some state that these rates include the commission, its value-added tax and the charges above [34]; we therefore set $c_{b} = 0.0015$ and $c_{s} = 0.0025$. A broker quoting these rates before fees would charge about 0.19% and 0.29%, a case covered by the cost-level sensitivity in Section 3.7. Costs are charged on traded value at execution (close of $t + 1$) and booked on the first holding day; spread, market impact and stamp duty are excluded (Section 4.2).

## 2.3 Order-2 log-signature features and Lévy areas

The remaining subsections follow the rebalancing chain of Section 1, from the information the model observes to the rule that turns beliefs into trades. Each stock's market history enters as a five-channel path: on every trading day the channels are $x = \left( \log O,\log H,\log L,\log C,\log(1 + V) \right)$, and for each window length $W \in \left\{ 5,20,60 \right\}$ days the most recent $W$ observations are joined into a piecewise-linear path $X:\lbrack 0,W\rbrack \rightarrow \mathbb{R}^{5}$. Its order-2 log-signature consists of the five increments $X_{W}^{i} - X_{0}^{i}$ and the ten Lévy areas [7]

$$
A^{ij} = \frac{1}{2}\int_{0}^{W}\left\lbrack \left( X_{s}^{i} - X_{0}^{i} \right)dX_{s}^{j} - \left( X_{s}^{j} - X_{0}^{j} \right)dX_{s}^{i} \right\rbrack,\quad 1 \leq i < j \leq 5 \tag{1}
$$

which for a piecewise-linear path is an exact finite sum over its segments. An increment records how far a channel moved over the window; a Lévy area records the order of the moves, since $A^{ij}$ is positive when moves in channel $i$ tend to precede moves in channel $j$ and negative in the opposite order. For the close–volume pair, its sign separates a volume surge that precedes a price move from one that follows it, which two paths with identical increments cannot tell apart. Each window yields 15 coordinates and the three windows 45 features, standardized with the mean and standard deviation of the training period only. The increments-only ablation LogSig-Inc keeps the 15 increments and drops the 30 Lévy areas (Section 2.8).

## 2.4 Multiscale gated network and composite loss

Beliefs about returns come from a compact multiscale gated network (MSGN). The 15 coordinates $z^{(W)}$ of each window pass through their own branch $h_{W} = \operatorname{ReLU}\left( U_{W}z^{(W)} + c_{W} \right) \in \mathbb{R}^{16}$, a gate $\omega = \operatorname{softmax}\left( U_{g}z + c_{g} \right)$ computed from all 45 features weights the three scales, and a linear head returns the forecast

$$
\hat{y} = a^{\top}\sum_{W}\omega_{W}h_{W} + b \tag{2}
$$

The network has 923 parameters shared across the 40 stocks, and its gate weights $\omega_{W}$ show which time scale drives the forecast (Section 3.1). The forecast five-day simple return is $\mu_{t,i} = \exp\left( {\hat{y}}_{t,i} \right) - 1$. Training minimizes a composite loss that balances calibration and cross-sectional ranking, in the spirit of losses designed against forecast collapse [13]:

$$
\mathcal{L}(\theta) = \operatorname{NMSE}(\theta) + \lambda_{rank}\frac{1}{\left| \mathcal{B} \right|}\sum_{t \in \mathcal{B}}\left( 1 - \rho\left( u_{t},v_{t} \right) \right) \tag{3}
$$

where NMSE is the mean squared error divided by the training variance of the target, $\rho$ is the Pearson correlation across stocks on date $t$, $u_{t}$ and $v_{t}$ are the cross-sectional z-scores of forecasts and targets clipped to $\lbrack - 3,3\rbrack$, $\mathcal{B}$ is a mini-batch of 32 trading days and $\lambda_{rank} = 1$. Optimization uses AdamW with weight decay $10^{- 4}$ for at most 30 epochs; learning rates $10^{- 3}$ and $3 \times 10^{- 4}$ are compared on validation, and early stopping with patience 8 keeps the checkpoint that minimizes $S = \mathcal{L}_{val} - 0.5\,\overline{IC} + 10\left( 1 - \overline{IC} \right)\mathbf{1}\left\{ \overline{IC} \leq 0 \right\}$, where $\overline{IC}$ is the mean validation rank IC and the last term rejects checkpoints whose rank IC is not positive. The selected weights are frozen, nothing is refitted in the test period, and every model is trained with five pinned seeds, 0 to 4.

## 2.5 Covariance estimation and the turnover-penalized mean-variance problem

Beliefs about risk come from the Ledoit–Wolf shrinkage estimator [11] applied to the previous 60 daily returns and scaled to the five-day horizon, $\Sigma_{t} = 5\,{\hat{\Sigma}}_{t}^{LW} + 10^{- 12}I$. The forecasts are shrunk toward their cross-sectional mean, ${\tilde{\mu}}_{t} = (1 - \alpha)\mu_{t} + \alpha{\bar{\mu}}_{t}\mathbf{1}$ with $\alpha = 0.5$. This keeps their ranking and halves their dispersion, and because a common term cancels under full investment it is equivalent to shrinking the forecast deviations toward zero; we disclose it because it narrows the gradient spread of Section 2.6 and so enters every trade decision. Given the weights $w^{-}$ held at the close of $t$, the portfolio solves

$$
\min_{w}\; - w^{\top}\tilde{\mu} + \frac{\gamma}{2}w^{\top}\Sigma w + \eta_{b}\mathbf{1}^{\top}\left( w - w^{-} \right)^{\text{+}} + \eta_{s}\mathbf{1}^{\top}\left( w^{-} - w \right)^{\text{+}} \tag{4}
$$

$$
\text{s.t.}\quad\mathbf{1}^{\top}w = 1,\quad 0 \leq w \leq w_{\max} \tag{5}
$$

with risk aversion $\gamma = 2.5$, cap $w_{\max} = 0.15$ and penalties $\eta_{b} = \kappa c_{b}$ and $\eta_{s} = \kappa c_{s}$ proportional to the purchase and sale costs of Section 2.2. The primary scale $\kappa = 5$ gives $\eta_{b} = 0.0075$ and $\eta_{s} = 0.0125$: penalizing turnover beyond the cost itself acts as a form of regularization that reduces the impact of estimation error [17, 18], and Section 3.7 varies $\kappa$. Writing $w - w^{-} = u^{\text{+}} - u^{-}$ with $u^{\text{+}},u^{-} \geq 0$ turns (4)–(5) into a convex quadratic program in 120 variables, solved by SLSQP with tolerance $10^{- 12}$ and at most 200 iterations.

## 2.6 Theorem 1: the no-trade test and its role

Whether the optimizer trades at date $t$ can be read from (4)–(5) before the problem is solved. Let $g = - \tilde{\mu} + \gamma\Sigma w^{-}$ be the gradient of the smooth part of the objective at the current weights, $J = \left\{ j:w_{j}^{-} > 0 \right\}$ the assets that can be sold and $I = \left\{ i:w_{i}^{-} < w_{\max} \right\}$ the assets that can be bought; both sets are non-empty because $\mathbf{1}^{\top}w^{-} = 1$ and $Nw_{\max} > 1$.

*Theorem 1 (no-trade test).* Let $w^{-}$ satisfy (5). Then holding $w^{-}$ is optimal for (4)–(5) if and only if

$$
\Delta g_{t} = \max_{j \in J}g_{j} - \min_{i \in I}g_{i} \leq \eta_{b} + \eta_{s} \tag{6}
$$

The proof in Appendix A writes the optimality condition with the subdifferential of the trading cost and eliminates the budget multiplier. Its economics is a single pair trade: selling a small amount $\epsilon$ of asset $j$ and buying the same amount of asset $i$ changes the smooth objective by $\epsilon\left( g_{i} - g_{j} \right)$ and the penalty by $\epsilon\left( \eta_{b} + \eta_{s} \right)$, so the trade pays exactly when $g_{j} - g_{i}$ exceeds the round-trip penalty, and the best pair sets the largest gradient among sellable assets against the smallest among buyable ones. The band $\eta_{b} + \eta_{s} = \kappa\left( c_{b} + c_{s} \right)$ equals 0.02 at $\kappa = 5$ and widens linearly in $\kappa$, so for given current weights a no-trade date at one penalty scale remains one at every larger scale.

Theorem 1 is not a new theorem. It is an explicit, multiplier-free specialization of standard subdifferential optimality conditions [22, 23] to full investment without a risk-free asset, long-only box limits and asymmetric costs, closely related to the no-trade characterizations in [20, 21], to the optimality conditions used by algorithms for transaction-cost portfolio problems [25] and to the violating-pair condition of SMO [26] (Appendix A.3). Its role here is diagnostic. At every rebalance date we record $\Delta g_{t}$, the margin $\Delta g_{t} - \left( \eta_{b} + \eta_{s} \right)$ and the assets attaining the maximum and minimum, and compare the verdict of (6) with the solver's, counting a date as traded when the solver's turnover $\lVert w_{t} - w_{t}^{-} \rVert_{1}$ exceeds a fixed tolerance from the configuration. Dates with a margin below $10^{- 6}$ in absolute value are reported separately as boundary cases, because there the verdict depends on solver tolerance. [CONFIRM: treatment of dates on which drift pushes a holding above $w_{\max}$, so that $w^{-}$ is infeasible and a trade is forced]

## 2.7 Signal-activation ratio $\mathbf{\psi}_{\mathbf{\kappa}}$: what it measures

Theorem 1 classifies single dates; the signal-activation ratio summarizes them over a backtest. For penalty scale $\kappa$ and the set $\mathcal{R}$ of rebalance dates,

$$
\psi_{\kappa} = \frac{1}{\left| \mathcal{R} \right|}\sum_{t \in \mathcal{R}}\mathbf{1}\left\{ \Delta g_{t} > \eta_{b} + \eta_{s} \right\} \tag{7}
$$

is the share of dates on which the gradient spread crosses the band, computed from the backtest run at that $\kappa$, because the current weights, and with them $\Delta g_{t}$, depend on the penalty. On its own $\psi_{\kappa}$ does not measure signal strength but the balance of forecast and risk pressure against cost: a collapsed forecast, ${\tilde{\mu}}_{i} \approx \bar{m}$ for every $i$, shifts all gradient components equally and cancels in the spread, leaving $\gamma\Sigma w^{-}$, which still moves as prices drift the weights. We therefore compare $\psi_{\kappa}$ with $\psi_{\kappa}^{0}$ from a risk-only benchmark: the same backtest with a constant forecast ($\mu = 0$, since any common value cancels) and the same $\gamma$, $\Sigma$, cap, penalties and costs, run once for each $\kappa$; at $\kappa = 5$ it coincides with the penalized GMV baseline of Section 2.8. The gap $\psi_{\kappa} - \psi_{\kappa}^{0}$, read as a curve over $\kappa$, is the forecast-collapse indicator. A gap near zero means that collapse has reached the trades, and a positive gap is the share of rebalance dates on which the forecast, not the risk model, carries the spread across the band; no cutoff is imposed.

## 2.8 Baselines and apples-to-apples protocol

All strategies share the covariance estimator, cap, costs, execution timing and initial portfolio, so that differences come from the beliefs or the decision rule alone. Four learned models share the composite loss, training protocol, seeds, $\alpha$ shrinkage and penalized problem of Sections 2.4 and 2.5: the full model, called LogSig below (MSGN on all 45 features); LogSig-Inc, the same architecture on the 15 increments (353 parameters), which isolates the Lévy areas; an MLP with one hidden layer of 32 units on the 45 features (1,505 parameters); and an LSTM with 32 hidden units on the raw sequence of the last 60 daily log returns (4,513 parameters), a learned model without path features. Three classical strategies complete the comparison. Hist-MV replaces the forecast with the mean of the previous 60 daily returns scaled to five days, shrunk with the same $\alpha = 0.5$, in two variants: unpenalized ($\eta_{b} = \eta_{s} = 0$, costs still charged) and with the primary penalty ($\eta_{b} = 0.0075$, $\eta_{s} = 0.0125$). GMV sets $\mu = 0$ and solves (4)–(5) with the same penalty, so it is minimum variance under the same friction; the unpenalized GMV appears only as a supplementary row. EW rebalances weekly to $1/N$ at the same costs, a benchmark that is difficult to beat out of sample [3].

## 2.9 Backtest protocol, two panels and metrics

Each strategy is simulated in one continuous run from 2 January 2024 to 2 October 2026, starting from $w^{-} = \mathbf{1}/N$ with the cost of the first rebalance charged. At the close of decision date $t$ the optimizer sees $w_{t}^{-}$, the holdings drifted to that close, and chooses the target $w_{t}$; the trade executes at the close of $t + 1$. Between rebalances the weights drift with prices,

$$
w_{u + 1,i} = \frac{w_{u,i}\left( 1 + R_{u,i} \right)}{\sum_{k}w_{u,k}\left( 1 + R_{u,k} \right)} \tag{8}
$$

where $R_{u,i}$ is the simple return of stock $i$ on day $u$, and the cost

$$
C_{t + 1} = c_{b}\mathbf{1}^{\top}\left( w_{t} - w_{t + 1}^{d} \right)^{\text{+}} + c_{s}\mathbf{1}^{\top}\left( w_{t + 1}^{d} - w_{t} \right)^{\text{+}} \tag{9}
$$

is charged against the holdings $w_{t + 1}^{d}$ drifted through day $t + 1$, the weights actually held at execution, and booked on the first holding day $t + 2$. The QP and the KKT log thus use the weights the optimizer saw, and the cost uses the weights the trade must move. [CONFIRM: on a no-trade date, whether no order is sent or the target $w_{t}$ is re-established against the drifted holdings at $t + 1$] The position is held over days $t + 2$ to $t + 6$, matching the forecast target of Section 2.1. Within each panel, rebased at its start, net daily returns give the net compound annual growth rate (CAGR), the annualized Sharpe ratio in excess of a constant risk-free rate of 6% per year over 252 trading days, the maximum drawdown (MDD) and the mean two-sided executed turnover $\bar{\tau}$, the average of $\lVert w_{t} - w_{t + 1}^{d} \rVert_{1}$ over rebalance dates. Headline tables report the mean and standard deviation across the five seeds; per-seed results are kept in the repository.

## 2.10 Statistical inference

Predictive content is the rank information coefficient (rank IC), the cross-sectional Spearman correlation between forecasts and realized five-day returns on each date, averaged over the panel. Because consecutive targets overlap, the one-sided t-test of a non-positive mean rank IC uses an effective sample size of $n/5$; it is applied per seed and to the seed-averaged forecast. Risk-adjusted performance against EW and GMV is the intercept of a regression of daily net returns on the benchmark's net returns, with Newey–West standard errors using Bartlett weights and $L = 5$ lags [35]. Because many configurations are evaluated, Sharpe ratios are deflated [32]: the deflated Sharpe ratio is the probability that the true Sharpe ratio exceeds the maximum expected among $M$ unskilled trials, adjusted for skewness, kurtosis and sample length, with the cross-trial variance of Sharpe ratios taken over the same panel.

We count $M = 1,235$ trials. The sensitivity grid crosses $\gamma \in \left\{ 1,2.5,5,10 \right\}$, $\kappa \in \left\{ 0,1,2.5,5,10 \right\}$ and a cost multiplier $m \in \left\{ 0.5,1,2 \right\}$ that scales costs and penalties together ($c_{b} = 0.0015m$, $c_{s} = 0.0025m$, $\eta_{b} = \kappa c_{b}$, $\eta_{s} = \kappa c_{s}$) for four models and five seeds (1,200 trials); the non-inclusive cost case of Section 2.2 ($c_{b} = 0.001933$, $c_{s} = 0.002933$) adds 20, the four classical strategies and the unpenalized GMV five, and the ten half-lives searched in Section 3.4, counted as selection, ten. The risk-only benchmark runs are diagnostics and are not counted. The headline deflated Sharpe ratio uses the seed-averaged return series, with the strategy-level value ($M = 7$) reported alongside. All parameters live in versioned configuration files, seeds and dependencies are pinned, and the raw data snapshot is archived as a versioned release asset whose SHA-256 hash every step verifies; one command, make all, regenerates every table and figure from it, and the repository README maps each table and figure to its source [CONFIRM: repository URL and data release tag].

# 3. RESULT AND DISCUSSION

## 3.1 Predictive content of the signal

We first establish how weak the signal is. Table 1 reports the mean rank IC of each forecasting strategy in both panels. In Panel A, LogSig attains [RESULT: mean rank IC of LogSig, Panel A; earlier run 0.0011] with a one-sided p-value of [RESULT: p-value; earlier run 0.47], and [RESULT: how many models reject a non-positive mean rank IC at 5%, per seed and seed-averaged]. [IF no model is significant: The forecasts are statistically indistinguishable from noise for every model, consistent with the very low signal-to-noise ratio of equity returns [12]; this is the weak-signal regime the paper studies.] / [IF some model is significant: The signal is weak but not absent for that model, and Sections 3.3–3.6 are read with this caveat.] The gate weights of LogSig average [RESULT: mean gate weights $\omega_{5}$, $\omega_{20}$, $\omega_{60}$], so [RESULT: which scale dominates and whether the weights vary across dates]. The rest of this section asks what the pipeline does with a signal of this strength.

**Table 1. Out-of-sample performance, Panels A and B (learned models: mean ± s.d. over five seeds)**

| Strategy | Net CAGR | Sharpe | MDD | Turnover | Rank IC | NW alpha vs EW | Deflated Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Panel A (test 2024–2025)** |  |  |  |  |  |  |  |
| LogSig | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| LogSig-Inc | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| MLP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Hist-MV, penalized | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Hist-MV, unpenalized | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GMV | [ ] | [ ] | [ ] | [ ] | – | [ ] | [ ] |
| EW | [ ] | [ ] | [ ] | [ ] | – | – | [ ] |
| GMV, unpenalized (supplementary) | [ ] | [ ] | [ ] | [ ] | – | [ ] | [ ] |
| **Panel B (full out-of-sample, incl. the 2026 continuation)** |  |  |  |  |  |  |  |
| LogSig | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| LogSig-Inc | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| MLP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Hist-MV, penalized | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Hist-MV, unpenalized | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GMV | [ ] | [ ] | [ ] | [ ] | – | [ ] | [ ] |
| EW | [ ] | [ ] | [ ] | [ ] | – | – | [ ] |
| GMV, unpenalized (supplementary) | [ ] | [ ] | [ ] | [ ] | – | [ ] | [ ] |


## 3.2 Out-of-sample performance, Panels A and B

Table 1 also compares all strategies on identical metrics. In Panel A, [RESULT: ranking by net Sharpe ratio, best and worst strategies with values]. Against EW the Newey–West alpha of LogSig is [RESULT: annualized alpha and t-statistic, Panel A], and against GMV [RESULT: alpha and t-statistic]; [RESULT: which strategies, if any, have alphas significant at 5% in either panel]. After deflation for $M = 1,235$ trials, [RESULT: deflated Sharpe ratios of LogSig and the best baseline, and whether any exceeds 0.95]. Panel B, which adds the 2026 continuation, [IF the ranking holds: preserves the ranking, so the result is not an artifact of the 2024–2025 window] / [IF NOT: changes the ranking as follows, so Panel A conclusions are window-specific]. Because the signal is weak, Table 1 is read less as a contest of forecasts than as a record of what each decision rule costs: [RESULT: turnover and cost drag of LogSig against MLP, LSTM and unpenalized Hist-MV].

## 3.3 Refusing to trade as a cost-consistent decision

Figure 1 plots the gradient spread $\Delta g_{t}$ of LogSig at each of the [RESULT: number of rebalance dates, Panel B] rebalance dates against the band $\eta_{b} + \eta_{s} = 0.02$ at $\kappa = 5$. The spread crosses the band on [RESULT: $\psi_{5}$ of LogSig, mean ± s.d. over seeds; earlier run 33.57%] of the dates, and on the rest the optimizer holds the current portfolio. Two pieces of evidence test whether this is the design succeeding rather than inertia. First, the rule must hold on real data: the verdict of Theorem 1 agrees with the solver's trade decision at [RESULT: agreement rate X, %] of rebalance dates over all optimizing strategies and seeds, with [RESULT: number of boundary cases] boundary cases (Table 3, Appendix A). Second, the declined trades are priced against the no-penalty counterfactual ($\kappa = 0$, same model, seed and costs): they would have earned [RESULT: net P&L of declined trades Y, bps per declined week and cumulative] after costs. [IF Y \< 0: Under a statistically weak signal, the optimizer's refusal to trade is not inertia but a cost-consistent decision: it rebalances only when the gradient spread exceeds the round-trip cost $\eta_{b} + \eta_{s}$, a rule that matches the solver at X% of rebalance dates, and the trades it declines would have lost Y basis points after costs.] / [IF Y is zero or positive: The rule matches the solver at X% of dates, but the declined trades would have gained Y basis points; the band is cost-consistent ex ante yet forgoes some profitable trades, which Section 4.2 treats as a limitation.]

[Figure 1: gradient spread $\Delta g_{t}$ against the band, drawn from the KKT log of the main run]

Figure 1. Gradient spread $\Delta g_{t}$ of LogSig at each rebalance date against the no-trade band $\eta_{b} + \eta_{s} = 0.02$ ($\kappa = 5$), Panel B; dates above the band are trading dates [CONFIRM: one seed shown or seed average]

The risk-only benchmark separates the forecast's share of these crossings. At $\kappa = 5$, $\psi_{5}^{0}$ is [RESULT: $\psi_{5}^{0}$], so the gap $\psi_{5} - \psi_{5}^{0}$ is [RESULT: gap at $\kappa = 5$]; Table 2 reports the curve over $\kappa$. [IF the gap is near zero at every $\kappa$: Forecast collapse has reached the trades, since risk pressure alone produces almost all band crossings.] / [IF the gap is positive: The forecast carries the spread across the band on that share of dates, so the weak signal still moves trades.]

## 3.4 Beyond smoothing: what Lévy areas add to turnover damping

Removing the Lévy areas changes how much the model trades: mean turnover moves from [RESULT: $\bar{\tau}$ of LogSig; earlier run 7.17%] for LogSig to [RESULT: $\bar{\tau}$ of LogSig-Inc; earlier run 37.99%] for LogSig-Inc, and MDD from [RESULT: earlier run 5.93%] to [RESULT: earlier run 8.99%] (Table 1). Temporal smoothing of the forecast is the baseline mechanism for such damping, because a smoother $\tilde{\mu}$ moves the gradient spread less from week to week; the question is whether Lévy areas also select which assets cross the band, and by how much. The week-on-week autocorrelation of the forecasts, averaged across stocks, is [RESULT: LogSig] for LogSig and [RESULT: LogSig-Inc] for LogSig-Inc. We smooth the LogSig-Inc forecasts with an exponentially weighted moving average whose half-life, chosen from ten candidates to match the LogSig autocorrelation, is [RESULT: selected half-life, days], rerun the backtest and define the selection share

$$
s = \frac{{\bar{\tau}}_{Inc,sm} - {\bar{\tau}}_{LogSig}}{{\bar{\tau}}_{Inc} - {\bar{\tau}}_{LogSig}} \tag{10}
$$

the part of the turnover gap that remains once smoothness is matched. Smoothing brings LogSig-Inc turnover to [RESULT: $\bar{\tau}$ of smoothed LogSig-Inc], so $s$ is [RESULT: X, %]. [IF s is clearly positive: Lévy areas damp turnover mainly by smoothing the forecast over time; beyond that, after matching forecast smoothness, they still account for X% of the turnover reduction by selecting which constituents cross the no-trade band.] / [IF s is near zero: Lévy areas damp turnover through smoothing alone; once smoothness is matched no reduction remains, and we claim no selection effect.]

## 3.5 Who triggers trades: concentration of band crossings across sectors and constituents

Each crossing has two triggers: the sellable asset with the largest gradient and the buyable asset with the smallest. Over the [RESULT: number of crossing dates, LogSig, $\kappa = 5$] crossing dates of the main run, Figure 2 maps these assets by constituent and by IDX-IC sector [CONFIRM: source document for the IDX-IC 2024 sector list, to be cited]. The Herfindahl index of trigger shares is [RESULT: Herfindahl across constituents] across constituents, against $1/40 = 0.025$ for uniform triggering, and [RESULT: Herfindahl across sectors] across sectors; the most frequent triggers are [RESULT: top trigger tickers and sectors with shares], with a trailing 60-day volatility of [RESULT: volatility of trigger assets against the other constituents]. [IF triggers spread across sectors and rotate over time: The weak signal fires rarely, but when it fires it rotates between sectors, so it still captures economic structure.] / [IF a few high-volatility names dominate: Crossings are driven by a few volatile names near the band edge rather than by sector information; this is a limitation (Section 4.2) and motivates asset-specific penalties (Section 4.3).]

[Figure 2: sector × constituent heatmap of trigger assets, drawn from the trigger files of the main run]

Figure 2. Trigger assets on crossing dates by IDX-IC sector and constituent (LogSig, $\kappa = 5$); each cell counts the dates on which the constituent has the largest gradient among sellable or the smallest among buyable assets

## 3.6 Weak-signal defense against Hist-MV and error maximization

Hist-MV is the classical setting for error maximization: without learned features, the optimizer maps sampling noise in historical means directly into weights [2]. Unpenalized, Hist-MV turns over [RESULT: $\bar{\tau}$ of unpenalized Hist-MV] per rebalance with an MDD of [RESULT: MDD of unpenalized Hist-MV]; the shared penalty brings this to [RESULT: $\bar{\tau}$ and MDD of penalized Hist-MV], and LogSig to [RESULT: $\bar{\tau}$ and MDD of LogSig] (Table 1). Because both variants use the same $\alpha$ shrinkage, $\Sigma$, cap and costs as LogSig, the first step isolates the penalty's share of the reduction and the second the share of the learned features: [RESULT: penalty share and feature share of the turnover and drawdown reduction, %; Sharpe ratios of the three]. [IF the penalty share dominates: Under weak signals the defense against error maximization is structural, carried by the penalty band rather than by the forecast.] / [IF the feature share dominates: The learned features, not only the penalty, protect against error maximization despite their weak predictive content.]

## 3.7 Sensitivity to the penalty scale $\mathbf{\kappa}$ ($\eta_{b} = \kappa \cdot 0.0015$, $\eta_{s} = \kappa \cdot 0.0025$), $\mathbf{\gamma}$ and transaction cost

Table 2 varies one setting at a time around the primary configuration ($\gamma = 2.5$, $\kappa = 5$, $m = 1$) for LogSig. Mean turnover [RESULT: whether ${\bar{\tau}}_{\kappa}$ decreases monotonically over $\kappa \in \left\{ 0,1,2.5,5,10 \right\}$, with values], as Theorem 1 suggests for a band that widens linearly in $\kappa$, although the theorem holds $w^{-}$ fixed while each run's holdings adapt to its own penalty. The gap $\psi_{\kappa} - \psi_{\kappa}^{0}$ [RESULT: shape of the gap curve over $\kappa$ and where, if anywhere, it vanishes]. Higher risk aversion [RESULT: effect of $\gamma$ on turnover and Sharpe ratio]. Halving or doubling costs and penalties together [RESULT: effect of $m$ on net CAGR and Sharpe ratio], and the non-inclusive case, about 1.22 times the primary all-in cost, [RESULT: net CAGR and Sharpe ratio relative to the primary configuration]. [IF conclusions are stable across the grid: The conclusions of Sections 3.3–3.6 do not hinge on the primary configuration.] / [IF NOT: The conclusions hold over part of the grid only, which bounds the claims of Section 4.1.]

**Table 2. Sensitivity of LogSig to $\kappa$, $\gamma$ and the cost level, one setting varied at a time (seed means) [CONFIRM: panel]**

| Setting | Net CAGR | Sharpe | MDD | Turnover | $\mathbf{\psi}_{\mathbf{\kappa}}$ | $\mathbf{\psi}_{\mathbf{\kappa}}^{\mathbf{0}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\kappa = 0$ (no penalty) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\kappa = 1$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\kappa = 2.5$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\kappa = 5$ (primary) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\kappa = 10$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\gamma = 1$ | [ ] | [ ] | [ ] | [ ] | [ ] | – |
| $\gamma = 5$ | [ ] | [ ] | [ ] | [ ] | [ ] | – |
| $\gamma = 10$ | [ ] | [ ] | [ ] | [ ] | [ ] | – |
| $m = 0.5$ | [ ] | [ ] | [ ] | [ ] | [ ] | – |
| $m = 2$ | [ ] | [ ] | [ ] | [ ] | [ ] | – |
| Non-inclusive costs ($c_{b} = 0.001933$, $c_{s} = 0.002933$) | [ ] | [ ] | [ ] | [ ] | [ ] | – |


# 4. CONCLUSION AND RECOMMENDATION

## 4.1 Conclusion

This study asked which components of a cost-aware mean-variance pipeline protect an investor from trading on a weak, learned signal, and answered it through the optimality conditions of the problem itself. First, the signal is [RESULT: verdict on predictive content, with rank IC and p-value]. Second, an explicit no-trade test, a specialization of standard KKT conditions to full investment, long-only box limits and asymmetric IDX costs, explains each rebalancing decision from the gradient alone and agrees with the solver at [RESULT: X]% of dates, and [RESULT: the declined trades would have lost or gained Y basis points after costs]. Third, the gap between the signal-activation ratio and its risk-only benchmark shows that [RESULT: the forecast drives a share of the crossings, or forecast collapse has reached the trades]. Fourth, Lévy areas reduce turnover [RESULT: mainly through smoothing with a selection share of X%, or through smoothing alone], and against Hist-MV [RESULT: penalty share against feature share]. [IF Sections 3.3–3.6 support it: The claim to remember is that under a weak signal the investor's defense is structural: the cost band read from the optimality conditions, not the forecast, decides when to trade, and it declines the trades that noise would have induced.]

## 4.2 Limitation

Some limitations bound the scope of these claims; others could threaten them. The universe requires a complete history to October 2026 and so conditions on survival, which may overstate return levels for all strategies alike; because the main claims compare decision rules on the same universe, survivorship bounds their scope more than it threatens them. A single chronological split with weights frozen after 2023 leaves the forecasts untested under refitting, and the declined-trade result of Section 3.3, the claim most exposed to sample noise, rests on [RESULT: number of declined weeks] declined weeks in one market. The weekly frequency, 40 large stocks, one exchange and a constant risk-free rate further limit generality. [IF Section 3.5 finds volatile-name triggers: Trade triggers concentrate on a few volatile names, so a uniform band treats unequal assets alike.]

The cost model is proportional and excludes the bid–ask spread, market impact, and the fixed Rp10,000 stamp duty on trade confirmations above Rp10 million; for a weekly rebalance of large-capitalization LQ45 constituents these are likely second order, but they raise the effective cost, so reported net returns are optimistic relative to live execution.

## 4.3 Recommendation

Three follow-ups would most directly strengthen or falsify the weak-signal-defense claim. For research, replicating the KKT diagnostic and the risk-only benchmark on other markets and frequencies, with rolling refits, would test whether the gap $\psi_{\kappa} - \psi_{\kappa}^{0}$ behaves the same way where signals are stronger. Replacing the uniform band with asset-specific or volatility-scaled penalties, or adapting $\kappa$ over time, would test whether the band can keep declining noise-driven trades without forgoing informative ones. Adding a quadratic impact term, under which the optimal policy trades only partially toward an aim portfolio [19], would show how a no-trade region and partial trading interact, and training the forecaster on the decisions it induces [16], for example on the KKT margin, is a natural next step. For practice, the test gives Indonesian investors a check that needs only the current weights, a forecast and a covariance matrix: before rebalancing, compare the largest gradient among holdings that can be sold with the smallest among assets that can be bought against the round-trip cost.

# DAFTAR PUSTAKA

[1] Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, *7*(1), 77–91. <https://doi.org/10.1111/j.1540-6261.1952.tb01525.x>

[2] Michaud, R. O. (1989). The Markowitz optimization enigma: Is 'optimized' optimal? *Financial Analysts Journal*, *45*(1), 31–42. <https://doi.org/10.2469/faj.v45.n1.31>

[3] DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *The Review of Financial Studies*, *22*(5), 1915–1953. <https://doi.org/10.1093/rfs/hhm075>

[4] Pemerintah Republik Indonesia. (1997). Peraturan Pemerintah Nomor 14 Tahun 1997 tentang Pajak Penghasilan atas Penghasilan dari Transaksi Penjualan Saham di Bursa Efek. <https://peraturan.bpk.go.id/Details/56237/pp-no-14-tahun-1997> [check: confirm official title and URL on the BPK page]

[5] PT Stockbit Sekuritas Digital. (2026). *Transaksi saham: Berapa biaya trading di Stockbit Sekuritas?* Pusat Bantuan Stockbit. Diakses 4 November 2026, dari <https://help.stockbit.com/id/article/transaksi-saham-berapa-biaya-trading-di-stockbit-sekuritas-1lbkyq9/>

[6] Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matemática Iberoamericana*, *14*(2), 215–310. <https://doi.org/10.4171/RMI/240>

[7] Chevyrev, I., & Kormilitzin, A. (2026). A primer on the signature method in machine learning. In C. Bayer, G. dos Reis, B. Horvath, & H. Oberhauser (Eds.), *Signature methods in finance* (pp. 3–64). Springer. <https://doi.org/10.1007/978-3-031-97239-3_1>

[8] Cuchiero, C., & Möller, J. (2025). Signature methods in stochastic portfolio theory. *SIAM Journal on Financial Mathematics*, *16*(4), 1239–1303. <https://doi.org/10.1137/24M1700223>

[9] Futter, O., Horvath, B., & Wiese, M. (2023). *Signature trading: A path-dependent extension of the mean-variance framework with exogenous signals* (arXiv:2308.15135). arXiv. <https://doi.org/10.48550/arXiv.2308.15135>

[10] Akyildirim, E., Gambara, M., Teichmann, J., & Zhou, S. (2025). Randomized signature methods in optimal portfolio selection. *Quantitative Finance*, 25(2), 197–216. <https://doi.org/10.1080/14697688.2025.2458613>

[11] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, *88*(2), 365–411. <https://doi.org/10.1016/S0047-259X(03)00096-4>

[12] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, *33*(5), 2223–2273. <https://doi.org/10.1093/rfs/hhaa009>

[13] Wan, S., Ma, M., Zhu, H., Liu, G., Wang, S., Wen, Q., & Liu, H. (2026). *Forecast collapse in time-series foundation models* (arXiv:2608.14106). arXiv. <https://doi.org/10.48550/arXiv.2608.14106>

[14] Goldfarb, D., & Iyengar, G. (2003). Robust portfolio selection problems. *Mathematics of Operations Research*, *28*(1), 1–38. <https://doi.org/10.1287/moor.28.1.1.14260>

[15] DeMiguel, V., Garlappi, L., Nogales, F. J., & Uppal, R. (2009). A generalized approach to portfolio optimization: Improving performance by constraining portfolio norms. *Management Science*, *55*(5), 798–812. <https://doi.org/10.1287/mnsc.1080.0986>

[16] Elmachtoub, A. N., & Grigas, P. (2022). Smart “predict, then optimize”. *Management Science*, *68*(1), 9–26. <https://doi.org/10.1287/mnsc.2020.3922>

[17] Olivares-Nadal, A. V., & DeMiguel, V. (2018). Technical note—A robust perspective on transaction costs in portfolio optimization. *Operations Research*, *66*(3), 733–739. <https://doi.org/10.1287/opre.2017.1699>

[18] Hautsch, N., & Voigt, S. (2019). Large-scale portfolio allocation under transaction costs and model uncertainty. *Journal of Econometrics*, *212*(1), 221–240. <https://doi.org/10.1016/j.jeconom.2019.04.028>

[19] Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *The Journal of Finance*, *68*(6), 2309–2340. <https://doi.org/10.1111/jofi.12080>

[20] Dybvig, P. H., & Pezzo, L. (2019). Mean-variance portfolio rebalancing with transaction costs. *SSRN Electronic Journal*. <https://doi.org/10.2139/ssrn.3373329> [verify: check for a later journal version]

[21] Mei, X., DeMiguel, V., & Nogales, F. J. (2016). Multiperiod portfolio optimization with multiple risky assets and general transaction costs. *Journal of Banking & Finance*, *69*, 108–120. <https://doi.org/10.1016/j.jbankfin.2016.04.002>

[22] Rockafellar, R. T. (1970). *Convex analysis*. Princeton University Press. <https://doi.org/10.1515/9781400873173>

[23] Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press. <https://doi.org/10.1017/CBO9780511804441>

[24] Lobo, M. S., Fazel, M., & Boyd, S. (2007). Portfolio optimization with linear and fixed transaction costs. *Annals of Operations Research*, *152*(1), 341–365. <https://doi.org/10.1007/s10479-006-0145-1>

[25] Best, M. J., & Hlouskova, J. (2005). An algorithm for portfolio optimization with transaction costs. *Management Science*, *51*(11), 1676–1688. <https://doi.org/10.1287/mnsc.1050.0418>

[26] Keerthi, S. S., Shevade, S. K., Bhattacharyya, C., & Murthy, K. R. K. (2001). Improvements to Platt's SMO algorithm for SVM classifier design. *Neural Computation*, *13*(3), 637–649. <https://doi.org/10.1162/089976601300014493>

[27] Gunawan, A., Fajriyah, R., Bimakasa, M. A., & Untari, S. N. (2024). Analysis of portfolio formation on the LQ45 stocks index, using the Markowitz and single index models. *BAREKENG: Jurnal Ilmu Matematika dan Terapan*, *18*(4), 2363–2374. <https://doi.org/10.30598/barekengvol18iss4pp2363-2374>

[28] Oktavianus Yusan, B., & Riyadi, S. (2024). Portfolio optimization: Application and comparison of Markowitz model and single index model on LQ45 stocks in Indonesia Stock Exchange. *International Journal of Management Science and Application*, *3*(1), 57–85. <https://doi.org/10.58291/ijmsa.v3i1.210>

[29] Putra, R. A., & Nurmawati, E. (2024). Prediction-based stock portfolio optimization using Bidirectional Long Short-Term Memory (BiLSTM) and LSTM. *Scientific Journal of Informatics*, *11*(3), 609–620. <https://doi.org/10.15294/sji.v11i3.5941>

[30] Haryono, A. T., Sarno, R., & Sungkono, K. R. (2024). Stock price forecasting in Indonesia stock exchange using deep learning: A comparative study. *International Journal of Electrical and Computer Engineering*, *14*(1), 861–869. <https://doi.org/10.11591/ijece.v14i1.pp861-869>

[31] Indra, Supian, S., Sukono, Riaman, Saputra, M. P. A., Azahra, A. S., & Pirdaus, D. I. (2025). Stock return prediction on the LQ45 market index in the Indonesia Stock Exchange using a machine learning algorithm based on technical indicators. *Journal of Risk and Financial Management*, *18*(12), 714. <https://doi.org/10.3390/jrfm18120714>

[32] Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, *40*(5), 94–107. <https://doi.org/10.3905/jpm.2014.40.5.094>

[33] PT Bursa Efek Indonesia. (2026). *Jam dan mekanisme perdagangan: Biaya transaksi di bursa*. Diakses 4 November 2026, dari <https://www.idx.co.id/id/produk/mekanisme-dan-jam-perdagangan>

[34] PT Ajaib Sekuritas Asia. (2026). *Biaya transaksi saham dan reksa dana*. Diakses 4 November 2026, dari <https://ajaib.co.id/biaya>

[35] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, *55*(3), 703–708. <https://doi.org/10.2307/1913610>

# APPENDIX A. KKT AND NORMAL-CONE DERIVATION

## A.1 Optimality via subdifferential and normal cones

Write (4)–(5) as the minimization of $q(w) + c(w)$ over $H \cap B$, where $q(w) = - w^{\top}\tilde{\mu} + \frac{\gamma}{2}w^{\top}\Sigma w$, $c(w)$ is the penalty term of (4), $H = \left\{ w:\mathbf{1}^{\top}w = 1 \right\}$ and $B = \left\{ w:0 \leq w \leq w_{\max} \right\}$. The problem is convex, $q$ is differentiable with $\nabla q(w^{-}) = g$, and $c$, $H$ and $B$ are polyhedral, so no further constraint qualification is needed and a feasible $w^{-}$ is optimal if and only if [22, 23]

$$
0 \in \nabla q(w^{-}) + \partial c(w^{-}) + N_{H}(w^{-}) + N_{B}(w^{-}) \tag{11}
$$

Here $\partial c(w^{-}) = \prod_{i}\left\lbrack - \eta_{s},\eta_{b} \right\rbrack$, because each coordinate of $c$ is $\eta_{b}x^{\text{+}} + \eta_{s}( - x)^{\text{+}}$ in $x = w_{i} - w_{i}^{-}$, whose subdifferential at $x = 0$ is $\left\lbrack - \eta_{s},\eta_{b} \right\rbrack$; $N_{H}(w^{-}) = \left\{ \lambda\mathbf{1}:\lambda \in \mathbb{R} \right\}$; and $N_{B}(w^{-})$ contains the vectors $\nu$ with $\nu_{i} \leq 0$ if $w_{i}^{-} = 0$, $\nu_{i} \geq 0$ if $w_{i}^{-} = w_{\max}$ and $\nu_{i} = 0$ otherwise.

## A.2 Elimination of the budget multiplier

Condition (11) holds if and only if there are $\lambda \in \mathbb{R}$ and, for every asset, $s_{i} \in \left\lbrack - \eta_{s},\eta_{b} \right\rbrack$ and $\nu_{i}$ as above with $g_{i} + s_{i} + \lambda + \nu_{i} = 0$. If $0 < w_{i}^{-} < w_{\max}$, then $\nu_{i} = 0$ and the condition reads $g_{i} - \eta_{s} \leq - \lambda \leq g_{i} + \eta_{b}$. If $w_{i}^{-} = 0$, then $\nu_{i} \leq 0$ absorbs any surplus and only $- \lambda \leq g_{i} + \eta_{b}$ remains; if $w_{i}^{-} = w_{\max}$, then $\nu_{i} \geq 0$ and only $g_{i} - \eta_{s} \leq - \lambda$ remains. The left inequality therefore binds every asset in $J$ and the right one every asset in $I$, so a multiplier exists if and only if

$$
\max_{j \in J}g_{j} - \eta_{s} \leq - \lambda \leq \min_{i \in I}g_{i} + \eta_{b} \tag{12}
$$

has a solution, that is, if and only if $\max_{J}g_{j} - \min_{I}g_{i} \leq \eta_{b} + \eta_{s}$, which is (6). The two costs enter only through their sum, and for fixed $w^{-}$ the condition is monotone in $\kappa$.

## A.3 Remark: link to the SMO violating pair

*Remark (link to sequential minimal optimization).* The condition in Theorem 1 shares the structure of the optimality check used in sequential minimal optimization (SMO) for support vector machines. Keerthi et al. [26, Lemma 1] note that a feasible $\alpha$ solves the SVM dual, a convex quadratic program with one linear equality constraint and box constraints, if and only if $b_{low} \leq b_{up}$, where $b_{low}$ is the largest and $b_{up}$ the smallest value of $F_{i} = w(\alpha) \cdot z_{i} - y_{i}$, a label-adjusted dual gradient, over two index sets determined by the bound status of each variable and its class label; an index pair with $F_{i} > F_{j}$ across these sets is said to define a violation [26, eq. 2.7], and optimality holds exactly when no such pair exists. In our problem, the budget constraint $\mathbf{1}^{\top}w = 1$ plays the role of the equality constraint, the assets that can be sold, $J = \left\{ j:w_{j}^{-} > 0 \right\}$, and the assets that can be bought, $I = \left\{ i:w_{i}^{-} < w_{\max} \right\}$, play the role of the two index sets, and $g = - \mu + \gamma\Sigma w^{-}$ is the gradient. The two conditions differ in the threshold. In SMO the exact condition carries no tolerance; the practical check $b_{low} \leq b_{up} + 2\tau$ adds a positive tolerance parameter $\tau$ because exact optimality is usually not attainable numerically [26, eq. 2.8], so $\tau$ is a stopping choice. In Theorem 1 the gap $\eta_{b} + \eta_{s}$ is not a tolerance: it is the exact width of the subdifferential of the trading cost at $w^{-}$, so the band holds at exact optimality and widens with $\kappa$. We therefore claim no novelty for the condition itself, which is a cost-penalized instance of standard KKT reasoning [22, 23]; what this paper adds is its use as a per-date diagnostic of the rebalancing decision (Section 3.3) and as the basis of the balance measure $\psi_{\kappa}$ (Section 2.7).

## A.4 Agreement between the test and the solver

Table 3 cross-tabulates the verdict of Theorem 1 against the solver's trade decision over all optimizing strategies, seeds, configurations and rebalance dates of the KKT log; boundary cases, with a margin below $10^{- 6}$ in absolute value, are listed separately.

**Table 3. Agreement between the verdict of Theorem 1 and the solver's trade decision**

| Theorem 1 verdict | Solver traded | Solver held | Total |
| --- | --- | --- | --- |
| Trade ($\Delta g_{t} > \eta_{b} + \eta_{s}$) | [ ] | [ ] | [ ] |
| Hold ($\Delta g_{t} \leq \eta_{b} + \eta_{s}$) | [ ] | [ ] | [ ] |
| Boundary (margin below $10^{- 6}$) | [ ] | [ ] | [ ] |
| Total | [ ] | [ ] | [ ] |

