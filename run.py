"""Point d'entrée : python run.py  (ouvre http://127.0.0.1:7860)"""
import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Local Image Qwen")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour accéder depuis un autre appareil du réseau")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    print(f"\n  Local Image Qwen  ->  http://{a.host}:{a.port}\n  (Ctrl+C pour arrêter)\n")
    uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
