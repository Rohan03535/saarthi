import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from Phase 4 Analysis
labels = ['High-Priority\nLeft Unattended\n(Score >= 2.0)', 'Low-Priority\nDispatched\n(Score < 1.0)']
bsl_values = [993, 1618]
opt_values = [810, 1444]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

rects1 = ax.bar(x - width/2, bsl_values, width, label='Naive Dispatch (Status Quo)', color='#da3633', alpha=0.9)
rects2 = ax.bar(x + width/2, opt_values, width, label='SAARTHI Dispatch', color='#58a6ff', alpha=0.9)

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Incidents (Nov 2023 - Apr 2024)', color='white', fontsize=12)
ax.set_title('The Saturation Hook: Who Gets Help When Units Run Out?', color='white', fontsize=14, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, color='white', fontsize=11)
ax.tick_params(axis='y', colors='white')
ax.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d', fontsize=10)

# Add values on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

plt.figtext(0.5, -0.05, 
            "SAARTHI rescued 241 critical chokepoints that Naive dispatch abandoned,\nby explicitly ignoring low-priority calls that Naive chased.",
            ha="center", fontsize=11, color="#3fb950", fontweight='bold',
            bbox={"facecolor": "#23863622", "edgecolor": "#3fb950", "pad": 10})

plt.tight_layout()
plt.savefig(r'C:\Users\Rohan\Pictures\saarthi\saarthi_impact_hook.png', dpi=150, bbox_inches='tight', facecolor="#0d1117")
print("Saved saarthi_impact_hook.png")
