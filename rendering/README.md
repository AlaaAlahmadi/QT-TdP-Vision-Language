# ECG rendering

`QT_Colouring_ECGRDVQ.ipynb` is the notebook that produced the images analysed in the paper
(`images/study/beat_colour` and `images/study/beat_line`, zipped as `Colour_Heartbeat.zip` and
`Colourless_Heartbeat.zip`, with `render_log_qtc.csv`). It runs in Google Colab:

1. **Signal processing (Python).** Lead II is read from each ECGRDVQ recording (1000 Hz) and preprocessed:
   0.5 Hz second-order Butterworth high-pass applied zero-phase (`scipy.signal.sosfiltfilt`), sym8 wavelet
   denoising (5 levels, soft universal threshold), NeuroKit2 `ecg_clean`, and subtraction of a 1001-sample
   median filter. R peaks are detected with a Pan–Tompkins-style pipeline (5–15 Hz band-pass, first difference,
   squaring, 120 ms moving-window integration, 200 ms minimum separation, prominence 0.2 × maximum, ±50 ms
   refinement), with NeuroKit2 `ecg_peaks` as a fallback.
2. **Colour vector.** For each sample, the colour index is `7 + (t - anchor)/40`, clipped to 1–9, where `t` is the
   time from Q onset (R peak − 28 ms) and `anchor` is the QT-nomogram threshold at the recording's mean heart rate.
3. **Drawing (R, run inside the notebook through `rpy2`).** `ggplot2` + `ggridges::geom_ridgeline_gradient`
   with the settings below.

| Setting | Value |
|---|---|
| Lead | II |
| Image size | 448 x 448 px (2.24 in at 200 dpi) |
| Time window | fixed 1600 ms (3.571 ms per pixel), opening 40 ms before the first R peak |
| Trace | drawn to 40 ms past the next R peak, blank afterwards; heartbeats with RR > 1520 ms are clipped (3 of 5,232) |
| Amplitude | -0.2 to 1.0 mV |
| Grid | 40 ms and 200 ms |
| Line width | 0.7, no axes, no padding (`expand = c(0, 0)`) |
| Colour scale | reversed Spectral palette; nine colour levels in 40 ms steps; level 7 = anchor |
| Coloured span | anchor - 240 ms to anchor + 80 ms (320 ms); at heart rate >= 85 bpm the upper limit is anchor + 40 ms (280 ms) |
| Outside the coloured span | filled grey (the scale's `na.value`) |
| Colourless images | the same trace as a line, with no fill |

## Python port

`render_beat_python.py` is a matplotlib reimplementation of the drawing step. It was not used for the images in
the paper. It is not pixel-identical to the R output, and its colour style leaves the area outside the coloured span
unfilled, whereas the R images fill it grey, so do not mix its images with the R-rendered ones.
