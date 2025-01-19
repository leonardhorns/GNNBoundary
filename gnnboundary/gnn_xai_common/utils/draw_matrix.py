import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import os

# sns.set_style("white")
sns.set(font_scale=1.6)


def draw_matrix(matrix, names, fmt='.2f', vmin=0, vmax=None,
                annotsize=20, labelsize=18, xlabel='Predicted', ylabel='Actual',
                max_label_length=15, labelpad=20, file_name=None, save_path=None):

    wrapped_names = [textwrap.fill(name, max_label_length) for name in names]
    label_factor = max(1, max(len(name) // max_label_length for name in names))
    plt.figure(figsize=(8 + label_factor * 2, 6 + label_factor * 2))

    ax = sns.heatmap(
        matrix,
        annot=True, annot_kws=dict(size=annotsize), fmt=fmt, vmin=vmin, vmax=vmax, linewidth=1,
        cmap=sns.color_palette("light:b", as_cmap=True),
        xticklabels=wrapped_names,
        yticklabels=wrapped_names,
    )
    ax.set_facecolor('white')

    ax.tick_params(axis='x', labelsize=labelsize, rotation=0)
    ax.tick_params(axis='y', labelsize=labelsize, rotation=0)

    if xlabel:
        plt.xlabel(xlabel, fontsize=labelsize, labelpad=labelpad)
    if ylabel:
        plt.ylabel(ylabel, fontsize=labelsize, labelpad=labelpad)

    plt.tight_layout()

    if file_name and save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path, dpi=300, format='jpg', bbox_inches='tight')
        print(f"Plot saved as: {full_path}")

    plt.show()