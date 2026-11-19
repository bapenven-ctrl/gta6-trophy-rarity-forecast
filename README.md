# GTA VI Trophy Rarity Forecast — a make.fun entry

Entry for the make.fun competition **gta6-trophy-rarity-forecast-01**: forecast,
for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the
players Exophase tracks for the game who will have earned it at
2026-11-26T00:00:00Z, as five quantiles per trophy (0.05, 0.25, 0.5, 0.75,
0.95), scored by mean pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads each case
file in the input directory — every read the house made of the game's
Exophase trophy page before the case's checkpoint — and writes one forecast
file per case to the output directory.

The forecast is a hybrid of three components, combined on the logit scale:

1. **Parametric earners/players decomposition.** A trophy's rate is
   earners / players_tracked, and the two move differently: the tracked pool
   grows (and each Exophase refresh can jump it, diluting every rate), while
   earners per trophy accumulate along a saturating curve. The program fits a
   growth model to players_tracked and a saturating curve to each trophy's
   earners over the case's distinct page states, then divides at the target.
2. **House-style curve.** A least-squares line of each trophy's logit rate
   against ln(1 + hours since launch), read at the target — the same family
   as the house curve baseline.
3. **Persistence.** The last read's rate, which anchors the ensemble when the
   recorded history is short or flat.

The spread around the ensemble median widens with the horizon, with the
distance the median moves from the last read, and with disagreement between
the three components; its scale was calibrated on a scenario ensemble
(simulated refresh, dilution and pool-growth patterns) to hold roughly 90%
empirical coverage of the 5–95% band. See `tools/validate.py` for the
validation harness and the numbers it produced.

The program is deterministic: two runs on the same case write the same bytes.

## Running it outside the grader

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
