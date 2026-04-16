# Gel-Surface Topography Reconstruction from Confocal Z-Stacks

This script reconstructs the 3D surface topography of gel samples (or similar soft materials) from fluorescence confocal microscopy z-stacks. It was built to handle a common problem in microscopy: figuring out exactly where the surface is when your signal is noisy and your z-resolution is limited.

Instead of relying on a single approach and hoping for the best, `integrated_analysis.py` runs two independent surface-finding methods, scores how trustworthy each one is at every pixel, and blends them together. The result is a height map that's detailed where the signal is strong and smooth where it isn't.

---

## What It Actually Does

You give it a `.tif` z-stack (the kind you'd export from a confocal microscope), and it gives you back a 32-bit TIFF height map in microns. Along the way, it does this:

### 1. Preprocessing

The raw stack goes through a few cleanup steps before anything else happens:

- **Background subtraction** — the minimum intensity along z is subtracted at each pixel. This gets rid of constant offsets from detector noise or autofluorescence.
- **Spatial binning** — optional downsampling in X/Y (controlled by `bin_factor`) to speed things up on large stacks.
- **Median filtering** — a small XY median filter on each slice to knock down single-pixel hot spots without blurring real features.

### 2. Method A — Sub-Pixel Argmax

The simplest idea: at each pixel, find which z-slice has the highest intensity. That's roughly where the surface is. But integer slice indices give you blocky results (your z-step might be 200 nm, and you want better than that), so after finding the peak slice, a three-point parabolic fit refines the position to sub-voxel precision:

```
δ = (I[k-1] − I[k+1]) / (2 · (2·I[k] − I[k-1] − I[k+1]))
```

This shifts the peak estimate by up to half a slice in either direction. It's fast, fully vectorized, and preserves fine spatial detail — but it's sensitive to noise. One noisy pixel can throw the local height off by several slices.

### 3. Method B — Patch-Based Spline (Schürmann et al.)

A more robust but lower-resolution approach. The image is divided into overlapping square patches (default 21×21 pixels). Within each patch, the intensity is averaged laterally to get a single clean z-profile, which is then upsampled with a cubic spline. The peak of the upsampled profile gives a sub-slice height estimate for that patch center.

Once all patch centers have height values, a 2D bicubic spline (`RectBivariateSpline`) interpolates them back to the full pixel grid. This gives a smooth surface that's resistant to noise, but it can't capture features smaller than the patch spacing.

### 4. Method C — The Integrated Surface (the main output)

This is where the two methods get combined. The idea is simple: use the argmax result where it's trustworthy, and fall back to the spline where it isn't. The blending weight comes from a per-pixel confidence score built from three metrics:

- **SNR** — signal-to-noise ratio of the z-profile peak relative to the background slices. High SNR means the argmax probably found a real surface, not noise.
- **Peak prominence** — how much the peak stands out above the mean intensity. A bright, well-defined peak is more likely to be the actual surface.
- **Spatial consistency** — how well the argmax height at a pixel agrees with its neighbors. An isolated spike is probably an artifact; a height that matches the local trend is probably real.

These three scores feed into a sigmoid-saturating combination. The sigmoid is tuned so that any pixel with decent SNR (above ~0.3 normalized) gets near-full argmax weight — the threshold is intentionally low because real surface pixels almost always clear it. A modulator built from prominence and consistency can reduce confidence by at most 25%, so genuinely bright beads or surface features don't get accidentally smoothed away.

The confidence map is Gaussian-blurred before blending to avoid sharp transitions between "argmax territory" and "spline territory." The final blend is:

```
integrated = w · argmax + (1 − w) · spline
```

After blending, a best-fit plane is subtracted (least-squares tilt removal) to correct for any sample or stage tilt, leaving you with relative surface deformation.

### 5. Interactive Line Profiling

After computing everything, the script opens a Matplotlib window showing the final height map. You can click two points on it to extract a cross-sectional height profile between them, plotted in microns. This is useful for quickly checking feature heights, groove depths, etc. without switching to another tool.

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- tifffile
- scikit-image

Install everything with:

```bash
pip install numpy scipy matplotlib tifffile scikit-image
```

---

## Usage

1. Open `surf7c.py` and set the `filename` variable to point at your `.tif` z-stack.
2. Adjust the physical parameters to match your microscope setup:
   - `nmppx` — nanometers per pixel in X/Y
   - `umPslice` — microns per z-slice
   - `bin_factor` — spatial downsampling factor (1 = no downsampling)
3. Optionally tweak the reconstruction parameters:
   - `patch_size` / `patch_overlap` — controls the spline method's spatial resolution
   - `z_upsample` — how finely the z-profiles are interpolated (higher = more precise but slower)
   - `CONF_SMOOTH_SIGMA` — blur radius on the confidence map
   - `SPATIAL_OUTLIER_SIGMA` — neighborhood size for the consistency check
4. Run it:

```bash
python integrated_analysis.py
```

The script will print progress to the console and save a `_integrated_surface.tif` file next to your input stack.

Set `DEBUG_FIGURES = True` (it's on by default) to see diagnostic plots showing each method's result, the confidence metrics breakdown, and height profile comparisons.

---

## Output

- **`<input_name>_integrated_surface.tif`** — a 32-bit float TIFF where each pixel value is the reconstructed surface height in microns, with tilt removed. You can open this in Fiji/ImageJ, Python, MATLAB, or any tool that reads float TIFFs.
- **Diagnostic plots** (when `DEBUG_FIGURES = True`):
  - A raw slice alongside the argmax height map
  - Side-by-side comparison of argmax vs. spline vs. integrated surfaces
  - The three confidence metrics and their combination
  - Median height profiles (with IQR bands) across all three methods
- **Interactive profile window** — click two points to get a cross-sectional height trace

---

## How the Confidence Blending Works (in plain terms)

The core challenge is that argmax gives you sharp detail but freaks out on noisy pixels, while the spline gives you a stable baseline but smears out small features. The confidence map essentially asks: "at this pixel, is there a clear enough signal that the argmax result is probably correct?"

If yes (strong fluorescence, well-defined peak, consistent with neighbors) → use argmax.
If no (dim signal, ambiguous peak, height doesn't match surroundings) → lean on the spline.

The sigmoid function makes this a soft decision rather than a hard cutoff, so you don't get visible boundaries between the two regimes.

---

## References

- The patch-based spline approach is adapted from: **Schürmann et al.** — the general idea of averaging within patches and fitting spline surfaces to the coarse grid comes from their work on gel surface reconstruction.
- The sub-pixel parabolic refinement is a standard technique in image processing for localizing peaks to sub-pixel accuracy.

---

## Notes

- The script currently expects a single-channel z-stack stored as a multi-page TIFF. If your data has multiple channels, extract the relevant one first.
- For very large stacks, increasing `bin_factor` to 2 or 4 can significantly speed things up at the cost of lateral resolution.
- The `z_upsample` parameter has diminishing returns past ~20. The original paper uses 100, but 20 is usually good enough and much faster.
- If your surface has features smaller than ~20 pixels across, consider reducing `patch_size` or increasing `patch_overlap` so the spline method can resolve them (though the argmax will likely handle them anyway if the SNR is decent).
