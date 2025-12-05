import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
import os


# --- KROK 1: MODYFIKACJA ŚRODOWISKA (WRAPPER) ---
class SpeedDemonWrapper(gym.Wrapper):
    """
    Modyfikacja środowiska "Pirat Drogowy".
    Agent dostaje dodatkową nagrodę TYLKO za utrzymywanie bardzo wysokiej prędkości.
    Ignorujemy standardowe nagrody za bezpieczną jazdę.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Wykonujemy krok w oryginalnym środowisku
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- MOJA MODYFIKACJA NAGRODY ---
        # Pobieramy prędkość z informacji zwrotnej (info)
        # W highway-env prędkość jest w m/s (zakładamy, że > 25 m/s to szybko)
        speed = info['speed']

        # Nowa funkcja nagrody:
        # Jeśli jedzie szybko (> 25 m/s) -> Duży bonus
        # Jeśli jedzie wolno -> Kara (ujemna nagroda)
        # Jeśli się rozbił (crashed) -> Ogromna kara

        custom_reward = 0
        if info['crashed']:
            custom_reward = -10.0
        elif speed > 25:
            custom_reward = 2.0  # Bonus za bycie piratem
        else:
            custom_reward = -0.5  # Kara za bycie "zawalidrogą"

        # Zwracamy zmodyfikowaną nagrodę
        return obs, custom_reward, terminated, truncated, info


# --- KROK 2: FUNKCJA TRENINGOWA ---
def train_agent():
    # Tworzymy folder na modele
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Tworzymy środowisko
    env_name = "highway-fast-v0"
    env = gym.make(env_name, render_mode="rgb_array")

    # 2. Nakładamy naszą modyfikację (Wrapper)
    env = SpeedDemonWrapper(env)
    print(f"Środowisko {env_name} uruchomione z modyfikacją 'SpeedDemon'.")

    # 3. Inicjalizujemy Agenta (PPO)
    # MlpPolicy to standardowa sieć neuronowa
    model = PPO("MlpPolicy", env, verbose=1)

    print("Rozpoczynam trening (to może chwilę potrwać)...")

    # Trenujemy przez 10,000 kroków (w prawdziwym projekcie daj więcej, np. 50k lub 100k)
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS)

    # Zapisujemy model
    model_path = f"{models_dir}/{env_name}_speed_demon"
    model.save(model_path)
    print(f"Model zapisany w: {model_path}")

    return model, env


# --- KROK 3: FUNKCJA TESTUJĄCA (WIZUALIZACJA) ---
def test_agent(model):
    print("Rozpoczynam test wizualny (Wydłużony)...")

    # Tworzymy środowisko
    env = gym.make("highway-fast-v0", render_mode="human")

    # --- NOWOŚĆ: KONFIGURACJA CZASU TRWANIA ---
    # Odwołujemy się do "wnętrza" środowiska (unwrapped), żeby zmienić ustawienia
    env.unwrapped.configure({
        "duration": 200,  # Czas trwania epizodu (zwiększony z 40 do 200!)
        "simulation_frequency": 15,  # Liczba klatek na sekundę (płynność)
        "policy_frequency": 1,  # Jak często agent podejmuje decyzję
    })

    # Ważne: Nakładamy Wrapper (musi być taki sam jak w treningu)
    env = SpeedDemonWrapper(env)

    obs, _ = env.reset()

    # Zmniejszyłem liczbę epizodów do 3, bo teraz będą dłuższe
    for i in range(10):
        done = False
        print(f"--- Epizod {i + 1} START ---")

        while not done:
            # Agent decyduje co zrobić
            action, _states = model.predict(obs, deterministic=True)

            # Wykonanie akcji
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()  # Pokaż na ekranie

            # Sprawdź czy koniec (wypadek lub koniec czasu)
            done = terminated or truncated

            # Jeśli był wypadek, wypisz info w konsoli
            if info.get('crashed'):
                print(f"Kraksa w epizodzie {i + 1}!")

        print(f"Epizod {i + 1} zakończony.")
        obs, _ = env.reset()

    env.close()


# --- GŁÓWNA PĘTLA ---
if __name__ == "__main__":
    # 1. Trenuj
    trained_model, training_env = train_agent()

    # 2. Testuj
    # Pytamy użytkownika, czy chce zobaczyć wynik
    odp = input("Trening zakończony. Czy chcesz zobaczyć jak jeździ agent? (t/n): ")
    if odp.lower() == 't':
        test_agent(trained_model)