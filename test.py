def test(env, num_episodes, load_path):
    wandb.init(
        entity="maytusp",
        # set the wandb project where this run will be logged
        project="test_single_dqn_foraging",
        name=f"config1",
        # track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "seq_length" : SEQ_LENGTH,
            "exploration_steps" : EXPLORE_STEPS,
            "buffer_size" : REPLAY_SIZE,
            "max_eps" : MAX_EPSILON,
            "min_eps" : MIN_EPSILON,
        }
    )
    agent = DQNAgent(ACTION_DIM, MESSAGE_DIM)
    # agent.q_network.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    agent.q_network.eval()
    epiode_rewards = []
    for episode in range(num_episodes):
        print(f"EPISODE: {episode}")
        frames = []
        obs = env.reset()
        image_seq = [obs['image'][0]]
        loc_seq = [obs['location'][0]]
        done = False
        total_reward = 0
        step = 0
        if VISUALIZE:
            print("START VISUALIZE")
            frame = visualize_environment(env, step)
            frames.append(frame.transpose((1, 0, 2)))
        while not done and step < 100:
            action, message = agent.select_action(image_seq[-SEQ_LENGTH:], loc_seq[-SEQ_LENGTH:], explore=False)
            env_action = env.int_to_act(action)
            next_obs, rewards, done, _, _ = env.step(env_action)

            image_seq.append(next_obs["image"][0])
            loc_seq.append(next_obs["location"][0])

            total_reward += sum(rewards)
            step += 1
            if VISUALIZE:
                frame = visualize_environment(env, step)
                frames.append(frame.transpose((1, 0, 2)))

        
        wandb.log(
            {
            "episode reward": total_reward}, step=episode
        )
        epiode_rewards.append(total_reward)
        if VISUALIZE:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_videofile(os.path.join(VIDEO_SAVED_DIR, f"ep_{episode}.mp4"), codec="libx264")
            
    wandb.log(
    {
    "average reward": total_reward}
    )  
    

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Environment()
# train_lstm_dql(env, 100000)
test(env, 100, "checkpoints/ckpt_3580000.pth")