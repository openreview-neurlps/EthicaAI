import traceback
from meltingpot import substrate

try:
    SUBSTRATE = 'commons_harvest__open'
    env_config = substrate.get_config(SUBSTRATE)
    roles = env_config.default_player_roles
    print("Roles found:", roles)
except Exception as e:
    traceback.print_exc()
