import os, sys, time, traceback, threading, webbrowser, socket
from urllib.request import Request, urlopen
import tkinter as tk
from PIL import Image, ImageTk


# -------------------------
# Logging (safe + robust)
# -------------------------

def log_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "BlueNuclei_logs", "logs")
    os.makedirs(d, exist_ok=True)
    return d

def log_path() -> str:
    return os.path.join(log_dir(), "BlueNuclei_launcher.log")

_log_lock = threading.Lock()

def write_log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        try:
            # Ensure directory exists even if called super early
            os.makedirs(os.path.dirname(log_path()), exist_ok=True)
            with open(log_path(), "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Change #1: DO NOT swallow silently — fall back to stderr
            try:
                sys.__stderr__.write(line)
                sys.__stderr__.flush()
            except Exception:
                pass

class _TeeStream:
    """Mirror writes to both original stream and the log file."""
    def __init__(self, original_stream, stream_name: str):
        self.original = original_stream
        self.stream_name = stream_name

    def write(self, s):
        try:
            if s:
                # Avoid spamming timestamps for every newline; still fine enough.
                for chunk in s.splitlines(True):
                    if chunk.strip():
                        write_log(f"{self.stream_name}: {chunk.rstrip()}")
        except Exception:
            pass

        try:
            self.original.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

def install_exception_hooks_and_stdio():
    # Change #3: capture stdout/stderr to log (and still forward to console when present)
    try:
        sys.stdout = _TeeStream(sys.__stdout__, "STDOUT")
    except Exception:
        pass
    try:
        sys.stderr = _TeeStream(sys.__stderr__, "STDERR")
    except Exception:
        pass

    # Change #2: capture uncaught exceptions (main thread)
    def excepthook(exc_type, exc, tb):
        write_log("=== Uncaught exception (sys.excepthook) ===")
        write_log("".join(traceback.format_exception(exc_type, exc, tb)))
        # Also show default behavior (prints to stderr)
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = excepthook

    # Change #2 (threads): Python 3.8+ has threading.excepthook
    if hasattr(threading, "excepthook"):
        def thread_excepthook(args):
            write_log("=== Uncaught exception in thread ===")
            write_log(f"Thread: {getattr(args, 'thread', None)}")
            write_log("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        threading.excepthook = thread_excepthook


# FORCE log creation immediately, but with safe fallback
write_log("=== launcher.py imported (very early) ===")


# -------------------------
# Resources + UI
# -------------------------

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

def show_splash(png_name: str = "splash.png"):
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    img_path = resource_path(png_name)
    pil_img = Image.open(img_path)
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

    def close():
        try: win.destroy()
        except Exception: pass
        try: root.destroy()
        except Exception: pass

    return root, close


# -------------------------
# Server helpers
# -------------------------

def wait_for_http(url: str, timeout_s: float = 180.0, interval_s: float = 0.2) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = Request(url, headers={"User-Agent": "BlueNuclei-Launcher"})
            with urlopen(req, timeout=1.0) as resp:
                code = resp.getcode()
                # treat anything that responds as "up"
                if 200 <= code < 500:
                    return True
        except Exception:
            time.sleep(interval_s)
    return False

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def run_flask_server(port: int):
    try:
        write_log("Importing server...")
        from app.server import app as flask_app
        write_log(f"Starting Flask on 127.0.0.1:{port} ...")
        flask_app.run(
            host="127.0.0.1",
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except Exception:
        write_log("=== Flask server failed to start ===")
        write_log(traceback.format_exc())

def try_shutdown_server(url_base: str, timeout_s: float = 2.0):
    """
    Change #4: attempt clean shutdown (requires /__shutdown route in server.py).
    If you don't add the route, this just fails harmlessly.
    """
    try:
        shutdown_url = url_base.rstrip("/") + "/__shutdown"
        req = Request(shutdown_url, headers={"User-Agent": "BlueNuclei-Launcher"})
        with urlopen(req, timeout=timeout_s):
            pass
        write_log("Shutdown request sent.")
    except Exception:
        write_log("Shutdown request failed or not supported (ok).")


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    install_exception_hooks_and_stdio()
    write_log("\n=== BlueNuclei launcher start ===")

    root, close_splash = show_splash("splash.png")

    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    write_log(f"Using port: {port}")

    server_thread = threading.Thread(target=run_flask_server, args=(port,), daemon=False)
    server_thread.start()

    # Make finish idempotent
    _finished = {"done": False}
    def finish():
        if _finished["done"]:
            return
        _finished["done"] = True

        # Change #4: best-effort clean shutdown; then don't hang forever.
        try_shutdown_server(url)

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

    root.after(50, lambda: threading.Thread(target=check_ready, daemon=True).start())

    root.mainloop()

    # Change #4: DO NOT join forever.
    # Give server a brief moment to stop; then exit anyway.
    write_log("Tk mainloop exited; waiting briefly for server thread.")
    server_thread.join(timeout=3.0)
    write_log("Launcher exit.")
