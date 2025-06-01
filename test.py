#!/usr/bin/env python3

import retro
import os
import json


def find_retro_data_locations():
    """Find where stable-retro stores data.json files"""

    print("🔍 Finding stable-retro data locations...\n")

    # Method 1: Get data path
    try:
        data_path = retro.data.get_data_path()
        print(f"📁 Main retro data path:")
        print(f"   {data_path}")
        print()
    except Exception as e:
        print(f"❌ Could not get data path: {e}")

    # Method 2: Check for SamuraiShodown-Genesis specifically
    game = "SamuraiShodown-Genesis"
    try:
        print(f"🎮 Checking for {game} data files...")

        # Try to get data.json path
        try:
            data_json_path = retro.data.get_file_path(game, "data.json")
            print(f"✅ data.json found:")
            print(f"   {data_json_path}")

            # Read and display the data.json content
            if os.path.exists(data_json_path):
                with open(data_json_path, "r") as f:
                    data_content = json.load(f)
                print(f"\n📄 Current data.json content:")
                print(json.dumps(data_content, indent=2))

        except Exception as e:
            print(f"❌ data.json not found: {e}")

        # Try to get other files
        for file_type in ["metadata.json", "scenario.json", "rom.sha"]:
            try:
                file_path = retro.data.get_file_path(game, file_type)
                print(f"✅ {file_type}: {file_path}")
            except:
                print(f"❌ {file_type}: Not found")

    except Exception as e:
        print(f"❌ Error checking {game}: {e}")

    print()

    # Method 3: List all available games and their data
    try:
        print("📋 Available games with data:")
        games = retro.data.list_games()

        # Look for fighting games or similar
        fighting_games = []
        for game in games:
            if any(
                term in game.lower()
                for term in ["fight", "street", "samurai", "mortal", "king"]
            ):
                fighting_games.append(game)

        print(f"\n🥊 Fighting games found ({len(fighting_games)}):")
        for game in fighting_games:
            print(f"   - {game}")

            # Check if this game has data.json
            try:
                data_path = retro.data.get_file_path(game, "data.json")
                print(f"     ✅ Has data.json: {data_path}")
            except:
                print(f"     ❌ No data.json")

    except Exception as e:
        print(f"❌ Could not list games: {e}")


def check_retro_installation():
    """Check retro installation details"""
    print("\n🔧 Retro installation info:")
    try:
        print(f"   Version: {retro.__version__}")
        print(f"   Location: {retro.__file__}")
    except:
        print("   Could not get version info")


def find_example_data_json():
    """Find an example data.json from any game"""
    print("\n📖 Looking for example data.json files...")

    try:
        games = retro.data.list_games()
        for game in games[:10]:  # Check first 10 games
            try:
                data_path = retro.data.get_file_path(game, "data.json")
                print(f"\n✅ Example from {game}:")
                print(f"   Path: {data_path}")

                # Show content
                with open(data_path, "r") as f:
                    content = json.load(f)

                if "info" in content:
                    print("   Info variables:")
                    for key, value in content["info"].items():
                        if isinstance(value, dict) and "address" in value:
                            print(f"     {key}: {value}")

                break  # Show just one example

            except:
                continue

    except Exception as e:
        print(f"❌ Could not find examples: {e}")


if __name__ == "__main__":
    print("🎌 Stable-Retro Data Location Finder\n")

    find_retro_data_locations()
    check_retro_installation()
    find_example_data_json()

    print("\n💡 Next steps:")
    print("1. If SamuraiShodown-Genesis has no data.json, you need to create one")
    print("2. Check other fighting games for memory address examples")
    print("3. Use memory scanning tools to find the correct addresses")
    print("4. Copy/modify an existing data.json as a template")
