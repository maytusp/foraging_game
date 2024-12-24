
import pygame


# Map keys to actions for each agent
KEY_MAPPING = {
    "agent1": {
        pygame.K_w: "up",
        pygame.K_s: "down",
        pygame.K_a: "left",
        pygame.K_d: "right",
        pygame.K_q: "pick_up",
        pygame.K_e: "idle",
    },
    "agent2": {
        pygame.K_UP: "up",
        pygame.K_DOWN: "down",
        pygame.K_LEFT: "left",
        pygame.K_RIGHT: "right",
        pygame.K_o: "pick_up",
        pygame.K_p: "idle",
    }
}

def get_agent_action(events, agent_id):
    # Check for key press events for the specific agent
    for event in events:
        if event.type == pygame.KEYDOWN:
            if agent_id == 0 and event.key in KEY_MAPPING["agent1"]:
                return KEY_MAPPING["agent1"][event.key]
            elif agent_id == 1 and event.key in KEY_MAPPING["agent2"]:
                return KEY_MAPPING["agent2"][event.key]
    return None
