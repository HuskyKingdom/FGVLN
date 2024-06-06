import matplotlib.pyplot as plt

# Re-defining data and settings with bolder lines
A_upstream = [0.94, 0.65, 0.76, 0.95, 0.78, 0.87]
A_downstream = [0.649, 0.801, 0.912, 0.984, 0.89, 0.925]
target_value_upstream = 0.865
target_value_downstream = 0.885
x_labels = ['Fst.', 'Snd.', 'Thd.', 'Lst.', 'Mid2', 'Mid3']

plt.figure(figsize=(12, 6))
# Upstream lines in blue shades, increased line width
plt.plot(x_labels, A_upstream, label='Downstream-FG-SR', marker='o', color='blue', linewidth=5.5)
plt.axhline(y=target_value_upstream, color='lightblue', linestyle='--', label='Downstream-GG-RandomShuffle', linewidth=5.5)

# Downstream lines in green shades, increased line width
plt.plot(x_labels, A_downstream, label='Upstream-FG-SR', marker='^', color='darkgreen', linewidth=5.5)
plt.axhline(y=target_value_downstream, color='lightgreen', linestyle='--', label='Upstream-GG-RandomShuffle', linewidth=5.5)

# Tight layout with optimized margins
plt.tight_layout(pad=2)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Hiding the top and right axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Setting labels and title
plt.xlabel('Negative Style',fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('SR of Fine-grained Predictions of Downstream R2R Trained SOTA Model', fontsize=20)
plt.ylim(0.5, 1.0)

# Adding legend
plt.legend(fontsize=18)

# Show the plot
plt.savefig("fg_test.pdf", format="pdf", dpi=600)
