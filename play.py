import retro
import pygame
import numpy as np
import sys
import time
import os
import gzip

# Constants - UPDATED FOR SAMURAI SHOWDOWN WITH STABLE-RETRO
GAME_NAME = "SamuraiShodown-Genesis"

# Initialize Pygame
pygame.init()

# Define window size and scaling factor
SCALE_FACTOR = 2
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 320, 224
WINDOW_WIDTH = ORIGINAL_WIDTH * SCALE_FACTOR
WINDOW_HEIGHT = ORIGINAL_HEIGHT * SCALE_FACTOR

# Create Pygame window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Samurai Showdown - Stable-Retro State Creator")

# Test different game names for stable-retro
print("🔍 Testing available Samurai Showdown games with stable-retro...")
alternative_names = [
    "SamuraiShodown-Genesis",
    "SamuraiShowdown-Genesis",
    "SamuraiShodown2-Genesis",
    "SamuraiShodown-Snes",
    "SamuraiShowdown-Snes",
    "SamuraiShodown-Nes",
    "SamuraiShowdown-Nes",
]

env = None
for alt_name in alternative_names:
    try:
        # stable-retro syntax
        env = retro.make(
            game=alt_name,
            state=retro.State.DEFAULT,  # Use DEFAULT state for stable-retro
            use_restricted_actions=retro.Actions.FILTERED,  # FILTERED is more common in stable-retro
            obs_type=retro.Observations.IMAGE,
            render_mode="rgb_array",
        )
        GAME_NAME = alt_name
        print(f"✅ Successfully loaded {alt_name} with stable-retro")
        break
    except Exception as e:
        print(f"❌ Failed to load {alt_name}: {e}")

if env is None:
    print("❌ Could not load any Samurai Showdown ROM")
    print("💡 Available games in stable-retro:")
    try:
        # List games available in stable-retro
        print("   Checking stable-retro data...")
        import retro.data

        games = retro.data.list_games()
        samurai_games = [
            g for g in games if "samurai" in g.lower() or "shodown" in g.lower()
        ]
        for game in samurai_games:
            print(f"   - {game}")
        if not samurai_games:
            print("   No Samurai Showdown games found")
            print("   Available games:")
            for game in sorted(games)[:10]:  # Show first 10 games
                print(f"   - {game}")
            print("   ...")
    except Exception as e:
        print(f"   Could not list games: {e}")
        print("   Make sure your ROM is properly imported")
    pygame.quit()
    sys.exit(1)

# Check available states for this game
print(f"🎯 Checking states for {GAME_NAME}:")
try:
    # In stable-retro, states might be handled differently
    states = retro.data.list_states(GAME_NAME)
    if states:
        for state in states:
            print(f"   - {state}")
    else:
        print("   Using default state")
except Exception as e:
    print(f"   Using default state (could not list: {e})")

# Print available buttons
print(f"🎮 Available buttons: {env.buttons}")

# Define key mappings
KEY_MAP = {
    pygame.K_UP: env.buttons.index("UP") if "UP" in env.buttons else None,
    pygame.K_DOWN: env.buttons.index("DOWN") if "DOWN" in env.buttons else None,
    pygame.K_LEFT: env.buttons.index("LEFT") if "LEFT" in env.buttons else None,
    pygame.K_RIGHT: env.buttons.index("RIGHT") if "RIGHT" in env.buttons else None,
    pygame.K_z: env.buttons.index("A") if "A" in env.buttons else None,
    pygame.K_x: env.buttons.index("B") if "B" in env.buttons else None,
    pygame.K_a: env.buttons.index("X") if "X" in env.buttons else None,
    pygame.K_s: env.buttons.index("Y") if "Y" in env.buttons else None,
    pygame.K_RETURN: env.buttons.index("START") if "START" in env.buttons else None,
    pygame.K_SPACE: env.buttons.index("SELECT") if "SELECT" in env.buttons else None,
    pygame.K_c: env.buttons.index("C") if "C" in env.buttons else None,
    pygame.K_d: env.buttons.index("Z") if "Z" in env.buttons else None,
    pygame.K_TAB: env.buttons.index("MODE") if "MODE" in env.buttons else None,
}

# Clean up None values
KEY_MAP = {k: v for k, v in KEY_MAP.items() if v is not None}

# Reset the environment
obs = env.reset()

# Track previous key states
previous_keys = {}
for key in list(KEY_MAP.keys()) + [pygame.K_t, pygame.K_h, pygame.K_o, pygame.K_p]:
    previous_keys[key] = False

# Health lock variables
health_lock_enabled = False
MAX_HEALTH = 100


def save_stable_retro_state(env, filename):
    """Save state compatible with stable-retro"""
    try:
        # For stable-retro, we may need to use different methods
        if hasattr(env, "em") and hasattr(env.em, "get_state"):
            # Method 1: Direct emulator state
            state_data = env.em.get_state()
        elif hasattr(env, "get_state"):
            # Method 2: Environment get_state
            state_data = env.get_state()
        else:
            print("❌ No get_state method found")
            return False

        # Save as gzipped file (stable-retro format)
        with gzip.open(filename, "wb") as f:
            f.write(state_data)

        print(f"✅ Stable-retro state saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving stable-retro state: {e}")

        # Try alternative method for stable-retro
        try:
            # Some versions of stable-retro might use different state handling
            env_state = (
                env.unwrapped.get_state()
                if hasattr(env.unwrapped, "get_state")
                else None
            )
            if env_state:
                with gzip.open(filename, "wb") as f:
                    f.write(env_state)
                print(f"✅ Alternative stable-retro state saved to {filename}")
                return True
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")

        return False


def load_stable_retro_state(env, filename):
    """Load stable-retro compatible state"""
    try:
        if not os.path.exists(filename):
            print(f"❌ State file not found: {filename}")
            return False

        with gzip.open(filename, "rb") as f:
            state_data = f.read()

        if hasattr(env, "em") and hasattr(env.em, "set_state"):
            env.em.set_state(state_data)
        elif hasattr(env, "set_state"):
            env.set_state(state_data)
        else:
            env.unwrapped.set_state(state_data)

        print(f"✅ Stable-retro state loaded from {filename}")
        return True
    except Exception as e:
        print(f"❌ Error loading stable-retro state: {e}")
        return False


# Print instructions
print("\n🎌 Samurai Showdown - Stable-Retro State Creator:")
print("- Arrow keys: Movement")
print("- Z/X/A/S: Attack buttons")
print("- Enter: START, Space: SELECT")
print("- T: Save TRAINING STATE (samurai_training.state)")
print("- O: Save temporary state")
print("- P: Load temporary state")
print("- H: Toggle health lock")
print("- ESC: Quit")
print("\n🎯 To create training states:")
print("1. Navigate to a challenging fight")
print("2. Press 'T' to save training state")
print("3. Use this state with your training script")

# State file paths
TRAINING_STATE_PATH = "samurai_training.state"
TEMP_STATE_PATH = "temp_samurai.state"

# Frame counter
frame_counter = 0

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    frame_counter += 1

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Get current key states
    current_keys = pygame.key.get_pressed()

    # Create action array
    action = [False] * len(env.buttons)

    # Set actions based on key mappings
    for key, button_idx in KEY_MAP.items():
        if current_keys[key]:
            action[button_idx] = True

    # Handle training state save
    if current_keys[pygame.K_t] and not previous_keys[pygame.K_t]:
        if save_stable_retro_state(env, TRAINING_STATE_PATH):
            print(f"🎯 TRAINING STATE saved as {TRAINING_STATE_PATH}")
            print(f"✅ Compatible with stable-retro training!")

    # Handle temporary state save/load
    if current_keys[pygame.K_o] and not previous_keys[pygame.K_o]:
        save_stable_retro_state(env, TEMP_STATE_PATH)

    if current_keys[pygame.K_p] and not previous_keys[pygame.K_p]:
        load_stable_retro_state(env, TEMP_STATE_PATH)

    # Toggle health lock
    if current_keys[pygame.K_h] and not previous_keys[pygame.K_h]:
        health_lock_enabled = not health_lock_enabled
        print(f"⚔️ Health lock {'enabled' if health_lock_enabled else 'disabled'}")

    # Update previous keys
    for key in previous_keys:
        previous_keys[key] = current_keys[key]

    # Step the environment
    try:
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        # Health lock implementation
        if health_lock_enabled and info:
            # Print available info keys to help debug
            if frame_counter % 300 == 0:  # Every 5 seconds
                print(f"🎮 Available info keys: {list(info.keys())}")

            # Try different health variable names
            health_vars = ["health", "hp", "life", "player_hp", "p1_health", "agent_hp"]
            for health_var in health_vars:
                if health_var in info and info[health_var] < MAX_HEALTH:
                    try:
                        # For stable-retro, health modification might work differently
                        if hasattr(env, "data"):
                            env.data.set_value(health_var, MAX_HEALTH)
                        elif hasattr(env.unwrapped, "data"):
                            env.unwrapped.data.set_value(health_var, MAX_HEALTH)
                        break
                    except Exception as e:
                        if frame_counter % 300 == 0:
                            print(f"❌ Could not set {health_var}: {e}")

    except Exception as e:
        print(f"❌ Error in env.step: {e}")
        break

    # Render the game
    try:
        frame = env.render()
        if frame is not None:
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(
                frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
    except Exception as e:
        print(f"❌ Error in rendering: {e}")
        break

    # Cap framerate
    clock.tick(60)

    # Reset if done
    if "done" in locals() and done:
        print("🎌 Resetting game...")
        obs = env.reset()

# Final instructions
print(f"\n🎯 Training state saved as: {TRAINING_STATE_PATH}")
print(f"✅ This state is stable-retro compatible!")

# Close everything
env.close()
pygame.quit()
sys.exit(0)
