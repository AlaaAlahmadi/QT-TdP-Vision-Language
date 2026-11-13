# ECG rendering

Place the scripts that turn ECGRDVQ recordings into the beat images here
(R code that produces `Colour_Heartbeat.zip`, `Colourless_Heartbeat.zip` and `render_log.csv`).

Rendering settings used in the paper:

| Setting | Value |
|---|---|
| Image size | 448 x 448 px (2.24 in at 200 dpi) |
| Time window | fixed 1600 ms (3.571 ms per pixel), opening 40 ms before the first R peak |
| Trace | drawn to 40 ms past the next R peak, blank afterwards; beats with RR > 1520 ms are clipped |
| Amplitude | -0.2 to 1.0 mV |
| Grid | 40 ms and 200 ms |
| Line width | 0.7, no axes, no padding (`expand = c(0, 0)`) |
| Lead | II |
| Colour scale | reversed Spectral palette; nine colour levels in 40 ms steps (eight intervals, 320 ms); level 7 = QT-nomogram threshold (anchor) for the recording's mean heart rate |
| Colour index | `7 + (t - anchor)/40`, clipped to 1-9; t = ms from Q onset (R peak - 28 ms) |
| Coloured span | anchor - 240 ms to anchor + 80 ms (320 ms); at heart rate >= 85 bpm the upper limit is anchor + 40 ms (280 ms, levels 1-8) |

## Python port

`render_beat_python.py` reimplements the R renderer in matplotlib (same window, grid, axis limits, palette and
gradient limits; ggplot2 line widths converted from mm to pt). It is not pixel-identical to the R output
(anti-aliasing and colour interpolation differ), so the images used in the paper are the R-rendered ones; do not mix
Python- and R-rendered images within one condition.

The input `mv` column is the fully preprocessed lead II signal (0.5 Hz zero-phase Butterworth high-pass, sym8 wavelet
denoising at the universal threshold, NeuroKit2 `ecg_clean`, subtraction of a 1001-sample median filter), the same
trace used for R-peak detection.
