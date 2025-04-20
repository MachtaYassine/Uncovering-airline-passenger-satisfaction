import os
import shutil

def main():
    mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'mlruns'))
    print(f"\033[91m[WARNING] This will delete all contents of {mlruns_path}. You will lose all your experiments and models.\n Use this command only if you want to reset your experiment folder completely.\033[0m")
    confirm = input("Are you sure you want to continue? [y/N]: ")
    if confirm.lower() == 'y':
        # Remove everything inside but keep the mlruns directory itself
        for item in os.listdir(mlruns_path):
            item_path = os.path.join(mlruns_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"All contents of {mlruns_path} have been deleted. Re-run your experiments")
    else:
        print("Aborted.")

if __name__ == "__main__":
    main()
