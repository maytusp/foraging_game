
import pygame


# Map keys to actions for each agent
KEY_MAPPING = {
    "agent1": {
        pygame.K_w: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
        pygame.K_d: 4,
        pygame.K_q: 5,
        pygame.K_e: 6,
    },
    "agent2": {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_o: 5,
        pygame.K_p: 6,
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
