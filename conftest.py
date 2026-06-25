"""pytest 루트 conftest — 저장소 루트를 sys.path에 추가해 `import src`가 되게 한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
