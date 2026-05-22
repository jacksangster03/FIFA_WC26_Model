from pathlib import Path
import shutil
import kagglehub

PROJECT_ROOT = Path("/Users/jack/FIFA_WC26_Model")
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
RAW_DIR = EXTERNAL_DIR / "international_results_raw"

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading latest Kaggle dataset...")

path = kagglehub.dataset_download(
    "martj42/international-football-results-from-1872-to-2017"
)

downloaded_path = Path(path)
print("KaggleHub cache path:", downloaded_path)

for file in downloaded_path.glob("*"):
    if file.is_file():
        dest = RAW_DIR / file.name
        shutil.copy(file, dest)
        print(f"Copied {file.name} -> {dest}")

results_file = RAW_DIR / "results.csv"
shootouts_file = RAW_DIR / "shootouts.csv"
goalscorers_file = RAW_DIR / "goalscorers.csv"

if results_file.exists():
    shutil.copy(results_file, EXTERNAL_DIR / "international_results.csv")
    print("Created data/external/international_results.csv")
else:
    raise FileNotFoundError("Could not find results.csv in Kaggle dataset.")

if shootouts_file.exists():
    shutil.copy(shootouts_file, EXTERNAL_DIR / "shootouts.csv")
    print("Created data/external/shootouts.csv")

if goalscorers_file.exists():
    shutil.copy(goalscorers_file, EXTERNAL_DIR / "goalscorers.csv")
    print("Created data/external/goalscorers.csv")

print("\nDone. Now run:")
print("python -m src.main --sim --n-sims 1000")
