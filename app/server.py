from flask import Flask, request, render_template, jsonify, send_file
import os, sys, traceback, tempfile, logging, json, time
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from .BlueNuclei_utils import process_single_image
from packaging.version import Version, InvalidVersion

def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static")
)

# load models
std_scaler = joblib.load(resource_path("std_scaler.pkl"))
minmax_scaler = joblib.load(resource_path("minmax_scaler.pkl"))
model = joblib.load(resource_path("svm_model.pkl"))

# --- version reading + update checking (same as you wrote) ---
_update_cache = {"t": 0.0, "msg": None}

def read_local_version(default="0.0.0") -> str:
    try:
        with open(resource_path("version.txt"), "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v if v else default
    except Exception:
        return default

VERSION = read_local_version()

def check_for_updates(cur_v):
    now = time.time()

    # cache result for 6 hours
    if now - _update_cache["t"] < 6 * 3600:
        return _update_cache["msg"]

    try:
        import requests
        repo_url = "https://github.com/ZHAN-ZHA/BlueNuclei"
        raw_url = "https://raw.githubusercontent.com/ZHAN-ZHA/BlueNuclei/main/version.txt"

        r = requests.get(raw_url, timeout=3)
        r.raise_for_status()
        latest = r.text.strip()

        try:
            if Version(latest) > Version(cur_v):
                msg = (
                    f"Current version: {cur_v}. "
                    f"Latest version: {latest}. "
                    f"Download it from: {repo_url}"
                )
            else:
                msg = None
        except InvalidVersion:
            # fallback: plain string compare if someone breaks version.txt
            msg = None if latest == cur_v else (
                f"Update available: {latest}. Download it from: {repo_url}"
            )

    except Exception:
        msg = None

    _update_cache["t"] = now
    _update_cache["msg"] = msg
    return msg

def safe_jsonify(obj):
    def default(o):
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, pd.Series):
            return o.to_dict()
        return str(o)
    return app.response_class(
        json.dumps(obj, default=default),
        mimetype='application/json'
    )

@app.route("/")
def index():
    message = check_for_updates(VERSION)
    return render_template("index.html", message=message)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        files = request.files.getlist("images")
        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(temp_dir, filename)
                file.save(temp_path)

                try:
                    result = process_single_image(temp_path, model, std_scaler, minmax_scaler)
                except Exception as e:
                    logging.error("Failed to process image", exc_info=True)
                    result = {"filename": filename, "status": f"❌ Error: {str(e)}"}
                results.append(result)

        # Convert any DataFrames inside results

        for i, val in enumerate(results):
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, pd.DataFrame):
                        results[i][k] = v.to_dict(orient="records")


        return safe_jsonify(results)

    except Exception as e:
        logging.error("Unexpected error during batch analysis", exc_info=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/temp_plot/<img_id>/<plot_name>')
def temp_plot(img_id, plot_name):
    temp_path = os.path.join(tempfile.gettempdir(), img_id, plot_name)
    return send_file(temp_path, mimetype='image/png')

@app.route('/visualize/<img_id>')
def visualize(img_id):
    return render_template("visualize.html", img_id=img_id)

@app.route('/diagram')
def diagram():
    return render_template('BlueNuclei_diagram.html')



