import random

templates = [
    "A video of an animal {action}. This action is a kind of {action_category} behavior, which is performed by the {segment}.",
]

def generate_prompt(data):
    action, action_category, segment = data['action'], data['action_category'], data['segment']
    return random.choice(templates).format(action=action, action_category=action_category, segment=segment)


if __name__ == "__main__":
    print(generate_prompt("running", "locomotion", "legs"))
