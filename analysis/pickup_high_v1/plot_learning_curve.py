import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
save_dir = "plots/lstm_vs_rmt_sp_15net/"
runs_dir = "../../runs/"
os.makedirs(save_dir, exist_ok=True)
model_name_list = [
                    # "lstm2k_sp_ppo_2net_invisible_wospeedrw", 
                    # "lstm67M_sp_ppo_2net_invisible_wospeedrw",
                    # "lstm_ppo_3net_invisible",
                    # "gpt_ppo_3net_invisible",
                    "lstm_ppo_15net_invisible",
                    "gpt_ppo_3net_invisible",
                    ]

name2config = {
                "lstm2k_sp_ppo_2net_invisible_wospeedrw": "grid13_img7_ni2_nw4_ms30_nwall13",
                "lstm67M_sp_ppo_2net_invisible_wospeedrw": "grid13_img7_ni2_nw4_ms30_nwall13",
                "lstm_ppo_3net_invisible": "grid5_img3_ni2_nw4_ms10",
                "gpt_ppo_3net_invisible": "grid5_img3_ni2_nw4_ms10",
}

name2legend = {
                "lstm2k_sp_ppo_2net_invisible_wospeedrw": "200k",
                "lstm67M_sp_ppo_2net_invisible_wospeedrw": "67M",
                "lstm_ppo_3net_invisible" : "LSTM",
                "gpt_ppo_3net_invisible" : "RMT",
                }
seed_list = [1, 2, 3]
num_net = 2
metrics = ['return', 'action_entropy', 'message_entropy']
colors = ['blue', 'green']

def extract_scalars(path, num_net):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={"scalars": 500}  # 0 = load all scalars
    )
    ea.Reload()
    # --- INSERT THIS TO INSPECT KEYS ---
    print(f"File: {path}")
    print("Available Scalar Keys:")
    episodic_return_tag = ""
    # ea.Tags() returns a dict with keys like 'images', 'scalars', 'histograms'
    for tag in ea.Tags()['scalars']: 
        if "episodic_return" in tag: # get either charts/episodic_return or charts/episodic_return/
            episodic_return_tag = tag
    # -----------------------------
    # Extract joint return
    return_events = ea.Scalars(episodic_return_tag)
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

for model_name in model_name_list:
    per_seed_data = {metric: [] for metric in metrics}

    for seed in seed_list:
        print(f"seed {seed}")
        config_name = name2config[model_name]
        folder = os.path.join(runs_dir, f"{model_name}/{config_name}_seed{seed}/")
        event_file = [f for f in os.listdir(folder) if f.startswith("events.out")][0]
        full_path = os.path.join(folder, event_file)

        steps, returns, a_entropy, m_entropy = extract_scalars(full_path, num_net)
        per_seed_data['return'].append(returns)
        per_seed_data['action_entropy'].append(a_entropy)
        per_seed_data['message_entropy'].append(m_entropy)

    for metric in metrics:
        all_data[metric][model_name] = {
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
    for i, model_name in enumerate(model_name_list):
        data = all_data[metric][model_name]
        plt.plot(data['steps'], data['mean'], label=name2legend[model_name], color=colors[i])
        plt.fill_between(data['steps'], data['mean'] - data['std'], data['mean'] + data['std'], color=colors[i], alpha=0.2)

    plt.title(f"")
    plt.xlabel("Steps")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{metric}_vs_steps.png")
    plt.savefig(save_path)
    plt.close()