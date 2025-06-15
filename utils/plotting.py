import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skfuzzy import control as ctrl
import numpy as np
from config import CRITERIA_COLORS, PLOT_FOLDER, CRITERIA_LABELS
import os


def plot_fuzzy_output_surface(score, margin, control_system, filename):
    """
    Generates a 3D surface plot for the fuzzy decision output over a range of scores and margins.
    """
    score_range = score.universe
    margin_range = margin.universe

    # Meshgrid for inputs
    x, y = np.meshgrid(score_range, margin_range)
    z = np.zeros_like(x)

    # Evaluate the fuzzy system across the grid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sim = ctrl.ControlSystemSimulation(control_system)
            sim.input['score'] = x[i, j]
            sim.input['margin'] = y[i, j]
            try:
                sim.compute()
                z[i, j] = sim.output['decision']
            except:
                z[i, j] = np.nan  # in case of no rule firing

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', linewidth=0.1)
    ax.set_title('Fuzzy Output Surface: Score vs. Margin')
    ax.set_xlabel('Score')
    ax.set_ylabel('Margin')
    ax.set_zlabel('Decision μ')
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{filename}_fuzzy_output_surface.png"))
    plt.close()


def plot_all_membership_functions(score, margin, decision, mu_cutoff):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    # ── Score ──
    for label in score.terms:
        axs[0].plot(score.universe, score[label].mf, label=label)
    axs[0].set_title("Score Membership Functions")
    axs[0].set_xlabel("Subtracted Cosine Score")
    axs[0].set_ylabel("Membership Degree")
    axs[0].legend()
    axs[0].grid(True)

    # ── Margin ──
    for label in margin.terms:
        axs[1].plot(margin.universe, margin[label].mf, label=label)
    axs[1].set_title("Margin Membership Functions")
    axs[1].set_xlabel("Margin (Best - Second Best)")
    axs[1].set_ylabel("Membership Degree")
    axs[1].legend()
    axs[1].grid(True)

    # ── Decision ──
    for label in decision.terms:
        axs[2].plot(decision.universe, decision[label].mf, label=label)
    axs[2].axvline(mu_cutoff, color='red', linestyle='--', label=f"μ cutoff = {mu_cutoff:.2f}")
    axs[2].set_title("Decision Membership Functions")
    axs[2].set_xlabel("Fuzzy Confidence (μ)")
    axs[2].set_ylabel("Membership Degree")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "all_membership_functions.png"))
    plt.close()
    
    
    
def plot_cosine_similarity_distribution(scores, thresholds, filename, bins=50, title="Subtracted Cosine Similarity Score Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, edgecolor='black')
    for i, threshold in enumerate(thresholds):
        if title == "Membership Degree Distribution":
            label = f'μ cutoff = {threshold}'
        else:
            label = f'{CRITERIA_LABELS[i]} ({threshold})'
        plt.axvline(x=threshold, color=CRITERIA_COLORS[i], linestyle='--', label=label)
    plt.title(title, fontsize=14)
    plt.xlabel("Subtracted Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if len(filename) > 70:
        filename = filename[:70] + "..."
    save_path = os.path.join(PLOT_FOLDER, f"{filename}_similarity_distribution.png")
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()


def plot_chunk_criteria_similarities(scores, criterion_label, chunk_idx, filename, chunk_text, threshold, i, top_n=10):
    plt.figure(figsize=(12, 6))
    
    plt.hist(scores, bins=40, alpha=0.5, label='Scores', color='blue')
    plt.axvline(x=threshold, color=CRITERIA_COLORS[i], linestyle='--', label=f'{CRITERIA_LABELS[i]} ({threshold})')
    
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Similarity Scores for Chunk {chunk_idx}\n{criterion_label}\n{chunk_text[:top_n]}...")
    plt.legend()
    plt.xlim(-0.05, 0.2)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    pic_filename = f"{filename[:50]}_chunk{chunk_idx}_similarity_histogram_{criterion_label}.png"
    plt.savefig(os.path.join(PLOT_FOLDER, pic_filename))
    plt.close()
