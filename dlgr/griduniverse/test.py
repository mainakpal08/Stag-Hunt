import os
import yaml

game_config_file = os.path.join(os.path.dirname(__file__), GAME_CONFIG_FILE)
with open(game_config_file, "r") as game_config_stream:
    game_config = yaml.safe_load(game_config_stream)
item_config = {o["item_id"]: o for o in game_config.get("items", ())}

# If any item is missing a key, add it with default value.
item_defaults = self.game_config.get("item_defaults", {})
for item in self.item_config.values():
    for prop in item_defaults:
        if prop not in item:
            item[prop] = item_defaults[prop]

hare_count = item_config.get("hare", {}).get("item_count", 0)
stag_count = item_config.get("stag", {}).get("item_count", 0)

