import random

templates = [
    "A video of an animal {action}. This action is a kind of {action_category} behavior.",
]

def generate_prompt(data):
    action, action_category = data['action'], data['action_category']
    return random.choice(templates).format(action=action, action_category=action_category)

if __name__ == "__main__":
    print(generate_prompt({"action": "running", "action_category": "locomotion"}))
