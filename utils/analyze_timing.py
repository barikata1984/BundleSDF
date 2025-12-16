import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def analyze_timing(csv_path, output_path):
    print(f"Reading CSV from: {csv_path}")
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Columns to plot
    cols = [
        'fm_prep_ms',
        'fm_2d_match_ms',
        'fm_corres_ms',
        'fm_ransac_ms',
        'bundle_adjust_ms',
        'others_ms'
    ]

    # Verify columns exist
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Columns not found in CSV: {missing_cols}")
        return

    print("-" * 60)
    print(f"{'Column':<20} | {'Mean':>10} | {'Median':>10} | {'99%tile':>10}")
    print("-" * 60)

    # Print total_ms if it exists
    if 'total_ms' in df.columns:
        col = 'total_ms'
        mean_val = df[col].mean()
        median_val = df[col].median()
        p99_val = df[col].quantile(0.99)
        print(f"{col:<20} | {mean_val:10.2f} | {median_val:10.2f} | {p99_val:10.2f}")
        print("-" * 60)

    for col in cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        p99_val = df[col].quantile(0.99)
        print(f"{col:<20} | {mean_val:10.2f} | {median_val:10.2f} | {p99_val:10.2f}")
    print("-" * 60)

    print("Plotting stacked bar chart...")
    # Plot stacked bar chart
    # Use row index as x-axis
    ax = df[cols].plot(kind='bar', stacked=True, figsize=(12, 6), width=1.0)
    
    # Customize plot
    plt.title('Timing Stats Stacked Bar Chart')
    plt.xlabel('Frame Index')
    plt.ylabel('Time (ms)')
    
    # Reduce x-tick clutter
    # If many frames, show fewer ticks
    n = len(df)
    if n > 50:
        ticks = ax.get_xticks()
        n_ticks = 10
        tick_spacing = max(1, int(len(ticks) / n_ticks))
        ax.set_xticks(ticks[::tick_spacing])
    
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save to file
    print(f"Saving plot to: {output_path}")
    try:
        plt.savefig(output_path)
        print("Save successful.")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Show popup
    # Check if running in a headless environment or explicitly disabled
    if os.environ.get('MPLBACKEND') == 'Agg':
        print("Skipping plt.show() because MPLBACKEND is Agg")
    else:
        print("Showing plot popup...")
        try:
            plt.show()
        except Exception as e:
            print(f"Error showing plot: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze timing statistics from a CSV file.")
    parser.add_argument("--csv-path", required=True, help="Path to the timing_stats.csv file")
    args = parser.parse_args()

    csv_path = args.csv_path
    
    # Derive output path: change extension to .png
    base, ext = os.path.splitext(csv_path)
    if ext.lower() != '.csv':
        print(f"Warning: Input file does not have .csv extension: {csv_path}")
    
    output_path = base + ".png"
    
    if not os.path.exists(csv_path):
        print(f"Error: Target CSV file not found at {csv_path}")
        sys.exit(1)
        
    analyze_timing(csv_path, output_path)
