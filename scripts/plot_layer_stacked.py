"""Stacked bar chart: domain composition of top neurons across normalized layers.
Aggregated across all 6 models."""

import numpy as np
import torch, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    ('Qwen_Qwen2-5-32B-Instruct', 'Qwen2.5-32B'),
    ('allenai_OLMo-2-0325-32B-Instruct', 'OLMo-32B'),
    ('mistralai_Mistral-Small-24B-Instruct-2501', 'Mistral-24B'),
    ('Qwen_Qwen2-5-72B-Instruct', 'Qwen2.5-72B'),
    ('meta-llama_Meta-Llama-3-1-70B-Instruct', 'Llama-70B'),
    ('mistralai_Mistral-Large-Instruct-2407', 'Mistral-123B'),
]
DOMAINS = ['Lan', 'MD', 'phys', 'ToM']
DOMAIN_DISPLAY = {'Lan': 'Language', 'MD': 'Formal', 'phys': 'Physics', 'ToM': 'Social'}
DOMAIN_COLORS = {'Lan': '#C0392B', 'MD': '#2471A3', 'phys': '#E67E22', 'ToM': '#27AE60'}
SHARED_COLOR = '#9B59B6'

pct = 0.1
N_BINS = 25


def load_domain_avg(results_dir, model_short, domain):
    domain_dir = os.path.join(results_dir, model_short, domain)
    if not os.path.isdir(domain_dir):
        return None
    all_scores = []
    for task in sorted(os.listdir(domain_dir)):
        path = os.path.join(domain_dir, task, 'neuron_attribution.pt')
        if os.path.exists(path):
            scores = torch.load(path, map_location='cpu').numpy()
            all_scores.append(scores)
    if not all_scores:
        return None
    return np.mean(all_scores, axis=0)


# Accumulate counts per bin: {category: [N_BINS]}
# Categories: Lan-only, MD-only, phys-only, ToM-only, shared
bin_counts = {d: np.zeros(N_BINS) for d in DOMAINS}
bin_counts['shared'] = np.zeros(N_BINS)
bin_model_count = np.zeros(N_BINS)  # how many models contribute to each bin

for model_short, model_name in MODELS:
    domain_attrs = {}
    for domain in DOMAINS:
        attr = load_domain_avg('results', model_short, domain)
        if attr is not None:
            domain_attrs[domain] = attr

    if len(domain_attrs) < 4:
        print(f'{model_name}: skipped')
        continue

    n_layers = domain_attrs[DOMAINS[0]].shape[0]
    n_neurons = domain_attrs[DOMAINS[0]].shape[1]

    # Get top neuron sets per domain (global threshold per domain)
    domain_top = {}
    for domain in DOMAINS:
        attr = domain_attrs[domain]
        threshold = np.percentile(np.abs(attr).flatten(), 100 - pct)
        domain_top[domain] = {}
        for layer in range(n_layers):
            domain_top[domain][layer] = set(np.where(np.abs(attr[layer]) >= threshold)[0])

    # Per layer: categorize neurons
    for layer in range(n_layers):
        norm_pos = layer / (n_layers - 1) if n_layers > 1 else 0
        bin_idx = min(int(norm_pos * N_BINS), N_BINS - 1)

        # Collect all top neurons across domains in this layer
        all_neurons = set()
        for d in DOMAINS:
            all_neurons |= domain_top[d][layer]

        for neuron in all_neurons:
            belongs_to = [d for d in DOMAINS if neuron in domain_top[d][layer]]
            if len(belongs_to) == 1:
                bin_counts[belongs_to[0]][bin_idx] += 1
            else:
                bin_counts['shared'][bin_idx] += 1

        bin_model_count[bin_idx] += 1

    print(f'{model_name}: {n_layers} layers processed')

# Average across models (divide by number of model-layers in each bin)
for key in bin_counts:
    for b in range(N_BINS):
        if bin_model_count[b] > 0:
            bin_counts[key][b] /= bin_model_count[b]

# Plot
# Log-transform each segment's count (not the axis).
# Each segment height = log10(count + 1). Stack linearly. Y axis is linear.
plot_counts = {k: np.log10(v + 1) for k, v in bin_counts.items()}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(7.68, 3.2))

x = np.arange(N_BINS)
width = 0.85
bottom = np.zeros(N_BINS)

# Stack order: Lan, MD, phys, ToM, shared
for domain in DOMAINS:
    ax.bar(x, plot_counts[domain], width, bottom=bottom,
           color=DOMAIN_COLORS[domain], label=DOMAIN_DISPLAY[domain],
           edgecolor='white', linewidth=0.3)
    bottom += plot_counts[domain]

ax.bar(x, plot_counts['shared'], width, bottom=bottom,
       color=SHARED_COLOR, label='Shared',
       edgecolor='white', linewidth=0.3)

# Labels
tick_positions = [0, N_BINS // 4, N_BINS // 2, 3 * N_BINS // 4, N_BINS - 1]
tick_labels = ['0%', '25%', '50%', '75%', '100%']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=11)
ax.set_xlabel('Model layer (normalized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Neuron count (log)', fontsize=12, fontweight='bold')
ax.set_title('Domain composition of task-critical neurons across layers\n(top 0.1%, averaged across 6 models)',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(loc='upper left', fontsize=15, frameon=False)
ax.set_xlim(-0.5, N_BINS - 0.5)

# Two-axis style (Science / PNAS): keep left + bottom only, thicker spines
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_linewidth(1.6)
ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.2,
               labelsize=11)

plt.tight_layout()

out_dir = 'results/figures'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f'{out_dir}/layer_stacked_domain_composition.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/layer_stacked_domain_composition.pdf', bbox_inches='tight')
print(f'\nSaved: {out_dir}/layer_stacked_domain_composition.png')
plt.close()
