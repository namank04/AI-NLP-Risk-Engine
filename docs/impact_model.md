# Impact score validation: event study

Does the engine's impact score anticipate how strongly a stock actually reacts?

- Unit: stock-day (a stock reacts once per day, however many articles). Train 2022-01 to 2023-06, test 2,141 stock-days from 2023-07-01, never seen in training.
- Label: |SCAR[0,+1]|, the 2-day abnormal return vs a SPY market model (220-day estimation window ending
  30 trading days before the event), in units of the stock's own residual volatility.

## Rank correlation with the realised reaction (test set)

Standard error of each ρ ≈ 0.022.

| predictor | Spearman ρ |
|---|---|
| learned | 0.072 |
| rule_v0 | 0.059 |
| news_count | 0.033 |
| max_abs_sentiment | 0.032 |
| p_earnings | 0.040 |

Paired bootstrap, learned minus rule: mean +0.013, 95% CI [-0.029, +0.057] → production uses **rule_v0**.

## Realised reaction by predicted decile (learned, test set)

|   decile |   n |   median_abs_car_pct |   mean_abs_scar |   share_2sigma |
|---------:|----:|---------------------:|----------------:|---------------:|
|        1 | 215 |               0.8731 |          0.6258 |         0.014  |
|        2 | 214 |               0.8339 |          0.6336 |         0.0327 |
|        3 | 214 |               0.9587 |          0.6542 |         0.0327 |
|        4 | 214 |               0.9977 |          0.7181 |         0.0327 |
|        5 | 214 |               0.9667 |          0.6999 |         0.0374 |
|        6 | 214 |               0.9832 |          0.7057 |         0.028  |
|        7 | 214 |               1.1423 |          0.793  |         0.0654 |
|        8 | 214 |               1.1582 |          0.7339 |         0.0514 |
|        9 | 214 |               1.3226 |          0.6887 |         0.0187 |
|       10 | 214 |               1.6777 |          1.0453 |         0.1542 |

## Same for the rule baseline (production impact score)

|   decile |   n |   median_abs_car_pct |   mean_abs_scar |   share_2sigma |
|---------:|----:|---------------------:|----------------:|---------------:|
|        1 | 215 |               0.9566 |          0.7099 |         0.0465 |
|        2 | 214 |               0.8736 |          0.615  |         0.0187 |
|        3 | 214 |               0.9339 |          0.6261 |         0.028  |
|        4 | 214 |               0.9388 |          0.6999 |         0.028  |
|        5 | 214 |               0.9935 |          0.771  |         0.0607 |
|        6 | 214 |               1.0207 |          0.7211 |         0.0607 |
|        7 | 214 |               1.3547 |          0.8407 |         0.0748 |
|        8 | 214 |               1.2682 |          0.7872 |         0.0327 |
|        9 | 214 |               1.1296 |          0.7581 |         0.0561 |
|       10 | 214 |               1.1053 |          0.7687 |         0.0607 |

## Does sentiment predict direction?

- Spearman(day mean sentiment, signed SCAR): 0.086
- per-signal sign hit rate when |sentiment| > 0.5: 0.527

## Feature importance (gain)

- p_earnings_max: 189.0
- vix: 183.7
- log_n: 104.2
- log_mcap: 66.7
- beta: 50.4
- rule_mean: 35.3
- vol20: 31.1
- sent_mean: 21.1
- abs_s_max: 21.0
- p_macroeconomic_max: 14.6
- p_operational_max: 12.9
- rule_max: 12.0
