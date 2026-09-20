# VaR Engine with Backtesting

## The finding

_Not established yet — this line gets replaced with the result, not with install
instructions._

The claim under test: **normal-distribution VaR under-predicts tail losses and its
exceptions cluster in high-volatility regimes; historical simulation adapts too slowly
to regime change; GARCH-based VaR passes conditional coverage where the others fail.**

The project is done when this README can answer, with a number and a chart:
_which of these models would you actually run, and how do you know?_

## What this is

A daily Value at Risk and Expected Shortfall engine for a ~10-position multi-asset
portfolio, computed three ways — historical simulation, parametric
variance-covariance, and Monte Carlo — over a hand-rolled GARCH(1,1) volatility model,
then backtested with Kupiec, Christoffersen, and Basel traffic-light tests.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data and stylized facts | not started |
| 2 | Historical + parametric VaR | not started |
| 3 | EWMA, GARCH(1,1) MLE, Student-t | not started |
| 4 | Monte Carlo | not started |
| 5 | Backtesting | not started |
| 6 | Stress testing + model documentation | not started |

## Layout

```
src/         tested, importable functions
notebooks/   orchestration and plots only
tests/       unit tests, including a no-lookahead test
docs/        model documentation memo
data/        cached price pulls (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
