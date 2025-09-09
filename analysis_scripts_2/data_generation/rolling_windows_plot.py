#!/usr/bin/env python3
"""
Rolling Windows Visualization - Complete Time Horizon with BlackRock Styling
Shows Windows 1-2 and final Windows with ellipsis in between
Time horizon: 2005 Q1 to 2023 Q3
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration parameters for full time horizon
start_year = 2005    # 2005 Q1 start
end_year = 2023      # 2023 Q3 end
end_quarter = 3      # Q3 of 2023

train_quarters = 20  # 5 years * 4 quarters
val_quarters = 2     # 6 months (2 quarters)
test_quarters = 2    # 6 months (2 quarters)

# Set exactly 27 windows as requested
total_windows = 26
total_quarters = (end_year - start_year) * 4 + end_quarter  # 2005 Q1 to 2023 Q3

# BlackRock professional styling
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 1,
})

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('white')

# BlackRock color palette
colors = {
    'training': '#003366',    # BlackRock Navy Blue
    'validation': '#FF8C00',  # Professional Orange
    'testing': '#228B22',     # Forest Green
    'text': '#333333'         # Dark gray text
}

# Display specific windows: 1, 2, 26, and 27
num_display_windows = 4
row_height = 1.0

# Calculate window positions for exactly 27 windows
windows_to_show = [
    (1, 0),  # Window 1, starting from beginning
    (2, test_quarters + 2),  # Window 2
    (25, None),  # Window 26 
    (26, None)   # Window 27 (final)
]

for display_idx, (window_num, offset) in enumerate(windows_to_show):
    if display_idx < 2:
        # First two windows - start from beginning
        window_start = offset if offset is not None else 0
    else:
        # Last two windows - calculate based on specific quarter requirements
        if display_idx == 2:  # Window 26
            # Window 26: Training 2017 Q2-2022 Q2, Validation 2022 Q3-2022 Q4, Test 2023 Q1-2023 Q2
            # Training starts at 2017 Q2 = (2017-2005)*4 + 1 = 49 quarters from start
            window_start = (2017 - start_year) * 4 + 1  # 2017 Q2
        else:  # Window 27
            # Window 27: Training 2017 Q3-2022 Q3, Validation 2022 Q4-2023 Q1, Test 2023 Q2-2023 Q3
            # Training starts at 2017 Q3 = (2017-2005)*4 + 2 = 50 quarters from start
            window_start = (2017 - start_year) * 4 + 2  # 2017 Q3
    
    # Reverse y-position calculation (top to bottom)
    y_base = (num_display_windows - 1 - display_idx) * row_height
    
    # Training rectangle
    rect_train = patches.Rectangle((window_start, y_base+0.1), train_quarters, 0.3,
                                   facecolor=colors['training'], alpha=0.7, 
                                   edgecolor='white', linewidth=0.5,
                                   label="Training (5Y)" if display_idx==0 else None)
    ax.add_patch(rect_train)
    
    # Validation rectangle
    rect_val = patches.Rectangle((window_start+train_quarters, y_base+0.1), val_quarters, 0.3,
                                 facecolor=colors['validation'], alpha=0.7,
                                 edgecolor='white', linewidth=0.5,
                                 label="Validation (6M)" if display_idx==0 else None)
    ax.add_patch(rect_val)
    
    # Testing rectangle
    rect_test = patches.Rectangle((window_start+train_quarters+val_quarters, y_base+0.1), test_quarters, 0.3,
                                  facecolor=colors['testing'], alpha=0.7,
                                  edgecolor='white', linewidth=0.5,
                                  label="Testing (6M)" if display_idx==0 else None)
    ax.add_patch(rect_test)
    
    # Window labels with professional styling (no "Final" labels)
    label = f"Window {window_num}"
    
    ax.text(window_start-2, y_base+0.25, label, va="center", ha="right", 
            fontsize=10, color=colors['text'], fontweight='bold')

# Add ellipsis in the middle
middle_y = (num_display_windows / 2 - 0.5) * row_height
middle_x = (windows_to_show[1][1] + train_quarters + val_quarters + test_quarters + 
           (total_quarters - (train_quarters + val_quarters + test_quarters))) / 2

ax.text(middle_x, middle_y + 0.25, "⋮\n⋮\n⋮", va="center", ha="center", 
        fontsize=16, color=colors['text'], fontweight='bold')

# Professional formatting with BlackRock styling
ax.set_xlim(-5, total_quarters + 5)
ax.set_ylim(-0.3, num_display_windows * row_height + 0.5)
ax.set_yticks([])

# Create x-axis labels showing years and quarters
x_ticks = range(0, total_quarters + 1, 8)  # Every 2 years
x_labels = []
for q in x_ticks:
    year = start_year + (q // 4)
    quarter = (q % 4) + 1
    x_labels.append(f"{year} Q{quarter}")

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Professional title with BlackRock styling (moved up to avoid overlap)
ax.set_title("Rolling Regression Illustration", fontsize=14, fontweight='bold', 
            pad=30, color=colors['text'])

# Add subtitle with time horizon
ax.text(0.5, 1.12, f"Time Horizon: {start_year} Q1 to {end_year} Q{end_quarter} | Total Windows: {total_windows}", 
        transform=ax.transAxes, ha='center', fontsize=10, color=colors['text'], style='italic')

# Professional legend styling (positioned lower to avoid title overlap)
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3,
                  framealpha=0.95, fancybox=True, shadow=True, borderpad=1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Style remaining spines
for spine in ['bottom']:
    ax.spines[spine].set_color('#CCCCCC')
    ax.spines[spine].set_linewidth(1)

# Set background color
ax.set_facecolor('#FAFAFA')

# Adjust layout to accommodate title and subtitle spacing
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()
