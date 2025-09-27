import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def leg_movement_data(file_path, sheet_name=None, leg=None):
    # Aim: always return three numeric arrays of length 100 (gait_phase, mean, std).
    # If the file or sheet cannot be read, or the sheet is shorter than expected,
    # return arrays padded with NaNs so callers (including cohort averaging) don't
    # fail due to None or inconsistent lengths.
    if leg is None:
        raise NotImplementedError("First select the leg (left or right)")
    leg = leg.lower()

    # default output length
    n_samples = 100

    # helper to build a NaN-filled fallback
    def nan_fallback():
        gait = np.arange(0, n_samples, dtype=float)
        mean = np.full(n_samples, np.nan, dtype=float)
        std = np.full(n_samples, np.nan, dtype=float)
        return gait, mean, std

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading '{file_path}' sheet '{sheet_name}': {e}")
        return nan_fallback()

    # determine rows to slice based on leg
    if leg == 'r':
        start, end = 0, 100
    elif leg == 'l':
        start, end = 102, 202
    else:
        print(f"Unknown leg '{leg}' provided. Expected 'l' or 'r'.")
        return nan_fallback()

    # safe slicing with iloc (may return fewer than n_samples rows)
    try:
        gait_series = pd.to_numeric(df.iloc[start:end, 0], errors='coerce')
        mean_series = pd.to_numeric(df.iloc[start:end, 1], errors='coerce')
        std_series = pd.to_numeric(df.iloc[start:end, 2], errors='coerce')
    except Exception as e:
        print(f"Error extracting columns from '{file_path}' sheet '{sheet_name}': {e}")
        return nan_fallback()

    # convert to numpy arrays and ensure length n_samples by padding with NaN if needed
    gait_vals = np.asarray(gait_series, dtype=float)
    mean_vals = np.asarray(mean_series, dtype=float)
    std_vals = np.asarray(std_series, dtype=float)

    # If gait column is completely missing or non-numeric, create a default axis
    if gait_vals.size == 0 or not np.any(np.isfinite(gait_vals)):
        gait_vals = np.arange(0, n_samples, dtype=float)

    # pad/truncate to n_samples
    def pad_or_truncate(arr):
        if arr.size >= n_samples:
            return arr[:n_samples]
        else:
            out = np.full(n_samples, np.nan, dtype=float)
            out[: arr.size] = arr
            return out

    gait_out = pad_or_truncate(gait_vals)
    mean_out = pad_or_truncate(mean_vals)
    std_out = pad_or_truncate(std_vals)

    return gait_out, mean_out, std_out

def plot_gait_motion(names, data_sheets, motion, leg=None, plot_std=True, cohort=False):
    """
    If cohort is False (default), behave as before: one figure per name.
    If cohort is True, produce a single figure with the averaged mean/std across all names.
    """
    # helper to parse sheet_add and side using a representative name (for comf special-case)
    def determine_sheet_add(data, representative_name, default_leg):
        str_split = data.split('_')
        side = None
        if str_split[1] == "comf":
            if representative_name == "anne":
                sheet_add = "0.25L"
                side = "L"
            else:
                sheet_add = str_split[1]
                side = "L"
                # side remains None; will fall back to default_leg if provided
        else:
            degree = float(str_split[2]) / 100
            degree_str = f"{degree:.2f}"
            side = str_split[3]
            sheet_add = degree_str + side

        # decide leg to use for reading
        leg_use = default_leg if default_leg is not None else side if side is not None else "L"
        return sheet_add, side, leg_use

    if not cohort:
        # Original behavior: a figure per name
        for name in names:
            print(f"===== {motion} variability {name} =====")
            plt.figure(figsize=(10, 6))
            for data in data_sheets:
                # parse sheet and decide leg without mutating the caller arg
                sheet_add, side, leg_use = determine_sheet_add(data, name, leg)

                gait, mean, std = leg_movement_data(file_path=name + data, sheet_name=motion + ' ' + sheet_add, leg=leg_use)

                # Alles naar numerieke numpy-arrays
                gait = np.asarray(gait, dtype=float)
                mean = np.asarray(mean, dtype=float)
                std = np.asarray(std, dtype=float)

                # Masker voor alleen eindige (niet-NaN, niet-inf) waarden
                mask = np.isfinite(gait) & np.isfinite(mean) & np.isfinite(std)

                # Plot mean
                plt.plot(gait[mask], mean[mask], label=f'Mean {motion} {sheet_add}')

                # Std-band (optioneel)
                if plot_std is True:
                    lower = (mean - std)[mask]
                    upper = (mean + std)[mask]
                    plt.fill_between(gait[mask], lower, upper, alpha=0.2)

                # Robuuste print van gemiddelde std
                print(f"{sheet_add}: {np.nanmean(std):.4f}")

            # --- Classic gait event markers (percent of gait cycle) ---
            gait_events = {
                'Foot flat': 8,
                'Midstance': 30,
                'Heel off': 40,
                'Toe off': 60,
                'Midswing': 80,
            }
            ax = plt.gca()
            for label, xpos in gait_events.items():
                ax.axvline(x=xpos, linestyle='--', linewidth=2, alpha=0.6)
                ax.annotate(
                    label,
                    xy=(xpos, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 6),
                    textcoords='offset points',
                    rotation=90,
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold'
                )
            ax.set_xlim(0, 100)

            plt.xlabel('Gait (%)')
            plt.ylabel(f'Mean {motion}')
            plt.title(f'Gait vs Mean {motion} {name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        # Cohort mode: one figure with averaged means/stds across all names
        print(f"===== {motion} cohort variability (averaged over {len(names)} subjects) =====")
        plt.figure(figsize=(10, 6))
        representative_name = names[0] if len(names) > 0 else ""
        for data in data_sheets:
            # determine sheet_add based on representative name (handles comf special-case)
            sheet_add, side, leg_use_default = determine_sheet_add(data, representative_name, leg)

            # collect arrays for each subject
            gait_list = []
            mean_list = []
            std_list = []
            for name in names:
                _, mean_sub, std_sub = leg_movement_data(file_path=name + data, sheet_name=motion + ' ' + sheet_add, leg=leg_use_default)
                # convert to numeric arrays (keep NaNs)
                mean_arr = np.asarray(mean_sub, dtype=float)
                std_arr = np.asarray(std_sub, dtype=float)
                # gait: try to get a consistent gait axis; prefer the first finite one
                gait_sub, _, _ = leg_movement_data(file_path=name + data, sheet_name=motion + ' ' + sheet_add, leg=leg_use_default)
                gait_arr = np.asarray(gait_sub, dtype=float)
                gait_list.append(gait_arr)
                mean_list.append(mean_arr)
                std_list.append(std_arr)

            if len(mean_list) == 0:
                continue

            # Use the first gait array that has finite values as x-axis
            gait_axis = None
            for g in gait_list:
                if np.any(np.isfinite(g)):
                    gait_axis = g
                    break
            if gait_axis is None:
                # fallback to a simple 0..N-1 axis if nothing is available
                gait_axis = np.arange(mean_list[0].shape[0], dtype=float)

            # Stack and compute elementwise averages ignoring NaNs
            means_stack = np.vstack([m for m in mean_list])
            stds_stack = np.vstack([s for s in std_list])

            mean_agg = np.nanmean(means_stack, axis=0)
            std_agg = np.nanmean(stds_stack, axis=0)  # approximate cohort std as mean of per-subject stds

            # Mask for plotting
            mask = np.isfinite(gait_axis) & np.isfinite(mean_agg)

            plt.plot(gait_axis[mask], mean_agg[mask], label=f'Cohort mean {motion} {sheet_add}')
            if plot_std:
                lower = (mean_agg - std_agg)[mask]
                upper = (mean_agg + std_agg)[mask]
                plt.fill_between(gait_axis[mask], lower, upper, alpha=0.2)

            print(f"{sheet_add}: cohort mean std {np.nanmean(std_agg):.4f}")

        # --- Classic gait event markers (percent of gait cycle) ---
        gait_events = {
            'Foot flat': 8,
            'Midstance': 30,
            'Heel off': 40,
            'Toe off': 60,
            'Midswing': 80,
        }
        ax = plt.gca()
        for label, xpos in gait_events.items():
            ax.axvline(x=xpos, linestyle='--', linewidth=2, alpha=0.6)
            ax.annotate(
                label,
                xy=(xpos, 0),
                xycoords=('data', 'axes fraction'),
                xytext=(0, 6),
                textcoords='offset points',
                rotation=90,
                ha='center', va='bottom', fontsize=9,
                fontweight='bold'
            )
        ax.set_xlim(0, 100)

        plt.xlabel('Gait (%)')
        plt.ylabel(f'Mean {motion}')
        plt.title(f'Gait vs Mean {motion} (Cohort average)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()