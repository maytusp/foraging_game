import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

num_net_list = [3, 6, 9, 12, 15]
seed_list = [1, 2, 3]
metrics = ['return', 'action_entropy', 'message_entropy']
colors = ['blue', 'green', 'red', 'purple', 'orange']
save_dir = "plots"
def extract_scalars(path, num_net):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={"scalars": 1000}  # 0 = load all scalars
    )
    ea.Reload()

    # Extract joint return
    return_events = ea.Scalars("charts/episodic_return")
    steps = np.array([e.step for e in return_events])
    returns = np.array([e.value for e in return_events])

    # Extract and average action and message entropy over network ids
    action_entropies = []
    message_entropies = []
    for nid in range(num_net):
        try:
            ae = ea.Scalars(f"agent{nid}/losses/action_entropy")
            me = ea.Scalars(f"agent{nid}/losses/message_entropy")
            action_entropies.append([e.value for e in ae])
            message_entropies.append([e.value for e in me])
        except:
            continue

    action_entropy = np.mean(np.array(action_entropies), axis=0)
    message_entropy = np.mean(np.array(message_entropies), axis=0)

    return steps, returns, action_entropy, message_entropy

# Store all data per metric and num_net
all_data = {metric: {} for metric in metrics}

for num_net in num_net_list:
    per_seed_data = {metric: [] for metric in metrics}

    for seed in seed_list:
        print(f"seed {seed}")
        folder = f"../runs/pop_sp_ppo_{num_net}net_invisible/grid5_img3_ni2_nw4_ms10_seed{seed}/"
        event_file = [f for f in os.listdir(folder) if f.startswith("events.out")][0]
        full_path = os.path.join(folder, event_file)

        steps, returns, a_entropy, m_entropy = extract_scalars(full_path, num_net)
        per_seed_data['return'].append(returns)
        per_seed_data['action_entropy'].append(a_entropy)
        per_seed_data['message_entropy'].append(m_entropy)

    for metric in metrics:
        all_data[metric][num_net] = {
            'mean': np.mean(per_seed_data[metric], axis=0),
            'std': np.std(per_seed_data[metric], axis=0),
            'steps': steps
        }

# Plotting
for metric in metrics:
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16
    })
    plt.figure(figsize=(10, 5))
    for i, num_net in enumerate(num_net_list):
        data = all_data[metric][num_net]
        plt.plot(data['steps'], data['mean'], label=f"{num_net} agents", color=colors[i])
        plt.fill_between(data['steps'], data['mean'] - data['std'], data['mean'] + data['std'], color=colors[i], alpha=0.2)

    plt.title(f"")
    plt.xlabel("Steps")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{metric}_vs_steps.pdf")
    plt.savefig(save_path)
    plt.close()