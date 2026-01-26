import threading, webbrowser
from urllib.request import urlopen
import tkinter as tk
from PIL import Image, ImageTk
import os, sys, time, traceback
import platform


def log_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "BlueNuclei", "logs")
    os.makedirs(d, exist_ok=True)
    return d

def log_path() -> str:
    return os.path.join(log_dir(), "BlueNuclei_launcher.log")

def write_log(msg: str):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# FORCE log creation immediately
write_log("=== launcher.py imported (very early) ===")


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def wait_for_http(url: str, timeout_s: float = 20.0, interval_s: float = 0.2) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.0):
                return True
        except Exception:
            time.sleep(interval_s)
    return False

def show_splash(png_name: str = "splash.png"):
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.withdraw()                 # keep root hidden
    root.overrideredirect(True)

    # Get screen size
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    img_path = resource_path(png_name)
    pil_img = Image.open(img_path)

    # SCALE to fit screen (max 60% of width/height)
    pil_img.thumbnail((int(sw * 0.3), int(sh * 0.3)))

    photo = ImageTk.PhotoImage(pil_img)
    w, h = pil_img.size
    x = (sw - w) // 2
    y = (sh - h) // 2

    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.geometry(f"{w}x{h}+{x}+{y}")

    label = tk.Label(win, image=photo, borderwidth=0, highlightthickness=0)
    label.image = photo
    label.pack()

    # IMPORTANT: do NOT call root.deiconify()

    def close():
        try: win.destroy()
        except Exception: pass
        try: root.destroy()
        except Exception: pass

    return root, close


def run_flask_server():
    from app.server import app as flask_app
    flask_app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)

from urllib.request import Request, urlopen

def wait_for_http(url: str, timeout_s: float = 180.0, interval_s: float = 0.2) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = Request(url, headers={"User-Agent": "BlueNuclei-Launcher"})
            with urlopen(req, timeout=1.0) as resp:
                code = resp.getcode()
                if 200 <= code < 500:
                    return True
        except Exception:
            time.sleep(interval_s)
    return False

def run_flask_server(port: int):
    try:
        write_log("Importing server...")
        from app.server import app as flask_app
        write_log(f"Starting Flask on 127.0.0.1:{port} ...")
        flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)
    except Exception:
        write_log("=== Flask server failed to start ===")
        write_log(traceback.format_exc())

def find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    write_log("\n=== BlueNuclei launcher start ===")

    root, close_splash = show_splash("splash.png")

    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    write_log(f"Using port: {port}")

    # Non-daemon server thread (critical)
    server_thread = threading.Thread(target=run_flask_server, args=(port,), daemon=False)
    server_thread.start()

    def finish():
        # close splash and exit Tk loop (run on Tk thread)
        try:
            close_splash()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass

    def check_ready():
        write_log(f"Waiting for server at {url} ...")
        ok = wait_for_http(url, timeout_s=180.0, interval_s=0.2)

        if ok:
            write_log("Server responded. Opening browser...")
            try:
                webbrowser.open(url)
                write_log("webbrowser.open() called.")
            except Exception:
                write_log("webbrowser.open() failed:")
                write_log(traceback.format_exc())
            root.after(0, finish)
        else:
            write_log("Timeout waiting for server.")
            try:
                import tkinter.messagebox as mb
                mb.showerror("BlueNuclei", f"Server did not start.\n\nLog:\n{log_path()}")
            except Exception:
                pass
            root.after(0, finish)

    # Start readiness check (non-daemon)
    root.after(50, lambda: threading.Thread(target=check_ready, daemon=False).start())

    # splash loop
    root.mainloop()

    # Keep interpreter alive; avoid "shutdown" threadpool errors later
    write_log("Tk mainloop exited; joining server thread.")
    server_thread.join()
