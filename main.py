import uvicorn
import sys
from pathlib import Path

# Añadir el directorio 'src' al path para poder importar api
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000, reload=False)
