from stable_baselines3.common.env_checker import check_env
from rl4salmon import Salmon, SalmonScipy

def test_Salmon():
    check_env(Salmon(), warn=True)
    check_env(SalmonScipy(), warn=True)