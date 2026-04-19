import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

print("Testing imports...")
try:
    from src import config
    print(f"✅ Config loaded. PROJECT_ROOT: {config.PROJECT_ROOT}")
    
    from src.data.data_ignition_improved import IgnitionDataset, PrecomputedIgnitionDataset
    print("✅ src.data.data_ignition_improved imported")
    
    from src.models.ignition import RobustFireIgnitionModel
    print("✅ src.models.ignition imported")
    
    print("\n🎉 Refactoring verification successful!")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    sys.exit(1)
