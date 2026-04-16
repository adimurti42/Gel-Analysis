import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile, imwrite
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from skimage.measure import profile_line
from scipy.interpolate import interp1d, RectBivariateSpline

filename = "INSERT_FILE_PATH"

nmppx = 544 / 3.0
umPslice = 0.2
bin_factor = 1

patch_size = 21
patch_overlap = 15
z_upsample = 20

CONF_SMOOTH_SIGMA = 3.0
SPATIAL_OUTLIER_SIGMA = 5
DEBUG_FIGURES = True


def compute_spline_surface(
    stack_xyz, um_per_slice, patch_size=21, patch_overlap=15, z_upsample=20
):

    nx, ny, nz = stack_xyz.shape
    step = patch_size - patch_overlap
    if step <= 0:
        raise ValueError("patch_size must be strictly larger than patch_overlap")

    x_centres = np.arange(patch_size // 2, nx, step)
    y_centres = np.arange(patch_size // 2, ny, step)
    nxc, nyc = len(x_centres), len(y_centres)
    coarse_height = np.zeros((nxc, nyc), dtype=np.float32)

    z_idx = np.arange(nz, dtype=np.float32)
    z_interp = np.linspace(0, nz - 1, nz * z_upsample, dtype=np.float32)

    for ix, cx in enumerate(x_centres):
        x0 = int(max(cx - patch_size // 2, 0))
        x1 = int(min(cx + patch_size // 2 + 1, nx))
        for iy, cy in enumerate(y_centres):
            y0 = int(max(cy - patch_size // 2, 0))
            y1 = int(min(cy + patch_size // 2 + 1, ny))

            prof = stack_xyz[x0:x1, y0:y1, :].mean(axis=(0, 1))

            if np.all(prof <= 0):
                coarse_height[ix, iy] = 0.0
                continue

            try:
                f = interp1d(z_idx, prof, kind="cubic")
                prof_interp = f(z_interp)
            except Exception:
                f = interp1d(z_idx, prof, kind="linear", fill_value="extrapolate")
                prof_interp = f(z_interp)

            coarse_height[ix, iy] = z_interp[np.argmax(prof_interp)] * um_per_slice

    if np.all(coarse_height == 0):
        return np.zeros((nx, ny), dtype=np.float32)

    spline2d = RectBivariateSpline(x_centres, y_centres, coarse_height)
    height_full = spline2d(
        np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32)
    )
    return height_full.astype(np.float32)


def subtract_best_fit_plane(height_map):
    nx, ny = height_map.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(nx * ny)]
    coeffs, _, _, _ = np.linalg.lstsq(A, height_map.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return (height_map - plane).astype(np.float32)


def compute_argmax_subpixel(stack_xyz, um_per_slice):
    nx, ny, nz = stack_xyz.shape
    peak_idx = np.argmax(stack_xyz, axis=2)  # (nx, ny) integer

    k = np.clip(peak_idx, 1, nz - 2)
    ix = np.arange(nx)[:, None]
    iy = np.arange(ny)[None, :]

    y0 = stack_xyz[ix, iy, k - 1]
    y1 = stack_xyz[ix, iy, k]
    y2 = stack_xyz[ix, iy, k + 1]

    denom = 2.0 * (2.0 * y1 - y0 - y2)
    delta = np.where(np.abs(denom) > 1e-9, (y0 - y2) / denom, 0.0)
    delta = np.clip(delta, -0.5, 0.5)

    boundary = (peak_idx == 0) | (peak_idx == nz - 1)
    delta = np.where(boundary, 0.0, delta)

    return ((peak_idx + delta) * um_per_slice).astype(np.float32)


def compute_pixel_confidence(
    stack_xyz,
    argmax_height_um,
    um_per_slice,
    spatial_sigma=5,
    sigmoid_k=10.0,
    sigmoid_mid=0.30,
):
    nx, ny, nz = stack_xyz.shape

    peak_val = np.max(stack_xyz, axis=2)
    n_bg = max(nz // 4, 1)
    sorted_z = np.sort(stack_xyz, axis=2)
    bg_mean = sorted_z[:, :, :n_bg].mean(axis=2)
    bg_std = sorted_z[:, :, :n_bg].std(axis=2) + 1e-6
    snr = (peak_val - bg_mean) / bg_std
    snr_75 = np.percentile(snr, 75)
    snr_norm = np.clip(snr / (snr_75 + 1e-6), 0.0, 1.0).astype(np.float32)

    mean_val = np.mean(stack_xyz, axis=2)
    prominence = np.clip((peak_val - mean_val) / (peak_val + 1e-6), 0.0, 1.0).astype(
        np.float32
    )

    local_smooth = gaussian_filter(
        argmax_height_um.astype(np.float32), sigma=spatial_sigma
    )
    local_dev = np.abs(argmax_height_um - local_smooth)
    dev_scale = um_per_slice * 4.0
    consistency = np.exp(-local_dev / dev_scale).astype(np.float32)

    snr_sig = 1.0 / (1.0 + np.exp(-sigmoid_k * (snr_norm - sigmoid_mid)))
    modulator = 0.75 + 0.15 * prominence + 0.10 * consistency
    confidence = (snr_sig * modulator).astype(np.float32)
    np.clip(confidence, 0.0, 1.0, out=confidence)

    return confidence, {
        "SNR (normalised)": snr_norm,
        "Peak prominence": prominence,
        "Spatial consistency": consistency,
    }


def compute_integrated_surface(
    argmax_height, spline_height, confidence, smooth_sigma=3.0
):
    weight = gaussian_filter(confidence.astype(np.float32), sigma=smooth_sigma)
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    integrated = weight * argmax_height + (1.0 - weight) * spline_height
    return integrated.astype(np.float32), weight


with TiffFile(filename) as tif:
    stack = np.stack([p.asarray() for p in tif.pages], axis=2).astype(
        np.float32, copy=False
    )

slices = stack.shape[2]

stack -= np.min(stack, axis=2, keepdims=True)

stack_resized = stack[::bin_factor, ::bin_factor, :]

filtered_stack = median_filter(stack_resized, size=(bin_factor, bin_factor, 1))

idx_size = filtered_stack.shape[:2]
x_um = np.array([0, idx_size[1]]) * (nmppx * bin_factor) / 1000.0
y_um = np.array([0, idx_size[0]]) * (nmppx * bin_factor) / 1000.0

print("Computing argmax surface …")
argmax_height = compute_argmax_subpixel(filtered_stack, umPslice)
idx = np.argmax(filtered_stack, axis=2)
med_idx = median_filter(idx, size=(5, 5))

print("Computing spline surface (this may take a moment) …")
height_spline = compute_spline_surface(
    filtered_stack,
    um_per_slice=umPslice,
    patch_size=patch_size,
    patch_overlap=patch_overlap,
    z_upsample=z_upsample,
)

print("Computing integrated surface …")

confidence, conf_metrics = compute_pixel_confidence(
    filtered_stack,
    argmax_height,
    umPslice,
    spatial_sigma=SPATIAL_OUTLIER_SIGMA,
)

integrated_abs, weight_map = compute_integrated_surface(
    argmax_height,
    height_spline,
    confidence,
    smooth_sigma=CONF_SMOOTH_SIGMA,
)

integrated = subtract_best_fit_plane(integrated_abs)

vmin_int = np.percentile(integrated, 2.5)
vmax_int = np.percentile(integrated, 97.5)

frac_argmax = float(weight_map.mean())
print(
    f"  Mean argmax weight : {frac_argmax:.2f}  "
    f"(fraction of surface dominated by argmax)"
)
print(f"  Integrated height range (2.5–97.5 %): " f"{vmin_int:.3f} – {vmax_int:.3f} µm")


out_dir = os.path.dirname(filename) or "."
basename = os.path.splitext(os.path.basename(filename))[0]

out_tiff = os.path.join(out_dir, basename + "_integrated_surface.tif")
imwrite(out_tiff, integrated)
print(f"Saved 32-bit TIFF: {out_tiff}")


if DEBUG_FIGURES:
    plt.close("all")

    plt.figure("D1 – Raw slice + argmax", figsize=(9, 5))
    mid_k = min(10, slices - 1)
    plt.subplot(1, 2, 1)
    plt.imshow(
        stack_resized[:, :, mid_k],
        extent=(x_um[0], x_um[1], y_um[1], y_um[0]),
        cmap="viridis_r",
    )
    plt.title("One raw slice")
    plt.colorbar()
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.imshow(
        argmax_height, extent=(x_um[0], x_um[1], y_um[1], y_um[0]), cmap="viridis_r"
    )
    plt.title("Argmax sub-pixel (absolute µm)")
    plt.colorbar(label="µm")
    plt.axis("equal")
    plt.tight_layout()

    fig_d2, axes_d2 = plt.subplots(1, 3, figsize=(16, 5), num="D2 – Method comparison")
    for ax, data, title in zip(
        axes_d2,
        [argmax_height, height_spline, integrated_abs],
        ["Argmax sub-pixel (abs. µm)", "Spline (abs. µm)", "Integrated (abs. µm)"],
    ):
        im = ax.imshow(
            data,
            extent=(x_um[0], x_um[1], y_um[1], y_um[0]),
            cmap="viridis_r",
            origin="upper",
        )
        ax.set_title(title)
        ax.axis("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="µm")
    plt.tight_layout()

    fig_d3, axes_d3 = plt.subplots(1, 4, figsize=(20, 5), num="D3 – Confidence metrics")
    items = list(conf_metrics.items()) + [("Combined confidence", confidence)]
    for ax, (name, arr) in zip(axes_d3, items):
        im = ax.imshow(
            arr,
            extent=(x_um[0], x_um[1], y_um[1], y_um[0]),
            cmap="plasma",
            vmin=0,
            vmax=1,
            origin="upper",
        )
        ax.set_title(name)
        ax.axis("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    smooth_sigma_plot = 1.0
    argmax_all = argmax_height
    spline_all = subtract_best_fit_plane(height_spline)
    integ_all = integrated

    plt.figure("D4 – Height profiles", figsize=(10, 5))
    x_px = np.arange(idx_size[1])
    for arr, label, color in [
        (argmax_all, "Argmax sub-pixel", "#1f77b4"),
        (spline_all, "Spline (detrended)", "#2ca02c"),
        (integ_all, "Integrated", "#d62728"),
    ]:
        med = gaussian_filter1d(np.median(arr, axis=0), sigma=smooth_sigma_plot)
        q25 = gaussian_filter1d(np.percentile(arr, 25, axis=0), sigma=smooth_sigma_plot)
        q75 = gaussian_filter1d(np.percentile(arr, 75, axis=0), sigma=smooth_sigma_plot)
        plt.plot(x_px, med, label=label, color=color, linewidth=2)
        plt.fill_between(x_px, q25, q75, color=color, alpha=0.15)
    plt.title("Height profiles: median ± IQR")
    plt.xlabel("X pixel (downsampled)")
    plt.ylabel("Height (µm)")
    plt.legend(fontsize="small")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

baseline = float(np.percentile(integrated, 2.5))
display_surface = np.clip(integrated - baseline, 0.0, None).astype(np.float32)

fig_ip, ax_ip = plt.subplots(1, 1, figsize=(7, 6))

vmin_disp = 0.0
vmax_disp = float(np.percentile(display_surface, 97.5))

im_ip = ax_ip.imshow(
    display_surface,
    cmap="viridis_r",
    vmin=vmin_disp,
    vmax=vmax_disp,
    origin="upper",
)
ax_ip.set_title(
    "Integrated surface  |  click two points to extract a line profile",
    fontsize=10,
    pad=15,
)
ax_ip.set_xlabel("Column (pixel)")
ax_ip.set_ylabel("Row (pixel)")
cb_ip = fig_ip.colorbar(im_ip, ax=ax_ip, fraction=0.046, pad=0.04)
cb_ip.set_label("Height (µm)")
fig_ip.tight_layout()

coords_ip = []


def extract_and_plot_profile(image, p0, p1, label="Integrated surface"):
    """Extract and plot a cross-sectional height profile (absolute height)."""
    prof = profile_line(
        image,
        (int(round(p0[1])), int(round(p0[0]))),
        (int(round(p1[1])), int(round(p1[0]))),
        linewidth=1,
    )

    px_spacing = nmppx * bin_factor / 1000.0  # µm per pixel
    dist_um = np.linspace(0, len(prof) * px_spacing, len(prof))

    fig_prof, ax_prof = plt.subplots(figsize=(7, 4))
    ax_prof.plot(dist_um, prof, linewidth=1.3, color="#d62728")
    ax_prof.set_title(f"Cross-section – {label}")
    ax_prof.set_xlabel("Distance (µm)")
    ax_prof.set_ylabel("Height (µm)")
    ax_prof.grid(True, alpha=0.3)
    fig_prof.tight_layout()
    plt.show()


def onclick(event):
    if event.inaxes is not ax_ip or event.xdata is None:
        return
    coords_ip.append((event.xdata, event.ydata))
    ax_ip.plot(event.xdata, event.ydata, "ro", markersize=5)
    fig_ip.canvas.draw()

    if len(coords_ip) == 2:
        p0, p1 = coords_ip
        extract_and_plot_profile(display_surface, p0, p1)
        coords_ip.clear()


cid = fig_ip.canvas.mpl_connect("button_press_event", onclick)
plt.show()
