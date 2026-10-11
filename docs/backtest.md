# Module A backtest (walk-forward, 2022-01 to 2023-12)

Portfolio formed at the close of D from signals dated D-1 or earlier; earns D+1. Transaction costs 5.0 bp per unit of one-way turnover; no-trade band 1%.

|                 |   ann_return |   ann_vol |   sharpe |   max_drawdown |   active_return |   tracking_error |   information_ratio |   active_t_stat |   avg_daily_turnover |
|:----------------|-------------:|----------:|---------:|---------------:|----------------:|-----------------:|--------------------:|----------------:|---------------------:|
| cap_weight      |       0.0222 |    0.2487 |   0.2124 |        -0.3007 |        nan      |         nan      |            nan      |        nan      |            nan       |
| equal_weight    |       0.0625 |    0.1926 |   0.4109 |        -0.2093 |          0.0263 |           0.1075 |              0.2447 |          0.3477 |            nan       |
| naive_tilt      |       0.0147 |    0.242  |   0.1811 |        -0.296  |         -0.009  |           0.0248 |             -0.3634 |         -0.5164 |              0.12048 |
| engine_tilt     |       0.0219 |    0.2407 |   0.2101 |        -0.2876 |         -0.0023 |           0.0175 |             -0.1301 |         -0.1849 |              0.03402 |
| black_litterman |       0.0352 |    0.2423 |   0.2637 |        -0.292  |          0.0111 |           0.0212 |              0.5217 |          0.7414 |              0.06825 |

## Signal quality: daily cross-sectional IC vs next-day abnormal return

|        |   mean_ic |   ic_t_stat |   hit_rate |   n_days |
|:-------|----------:|------------:|-----------:|---------:|
| engine |   -0.0244 |      -1.971 |     0.4695 |      509 |
| naive  |   -0.0226 |      -2.005 |     0.4558 |      509 |
