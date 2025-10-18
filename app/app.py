from flask import Flask, render_template, redirect, request, jsonify, session
from deepface import DeepFace
from scipy.spatial.distance import cosine
import cv2
import joblib
import numpy as np
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change-this-to-a-strong-secret"  # needed for session

# ========= Model =========
model = joblib.load("app/crop_model.pkl")

# ========= Face Embeddings =========
print("🧠 Encoding known faces...")
def load_known_faces():
    print("🔄 Loading known faces on demand...")
    faces = {}
    faces["Salim"] = DeepFace.represent(
        img_path="salim.jpg", model_name="VGG-Face", enforce_detection=False
    )[0]["embedding"]
    faces["Chedly"] = DeepFace.represent(
        img_path="chedly.jpg", model_name="VGG-Face", enforce_detection=False
    )[0]["embedding"]
    return faces

# Global variable (empty until loaded)
known_faces = None


threshold = 0.75

# ========= SQLite Logging =========
DB_PATH = "predictions.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            state_name TEXT,
            district_name TEXT,
            crop_year INTEGER,
            season TEXT,
            crop TEXT,
            area REAL,
            prediction REAL,
            created_at TEXT
        )
        """)
        conn.commit()

init_db()

def log_prediction(user, payload):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO logs (user, state_name, district_name, crop_year, season, crop, area, prediction, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user,
            payload["State_Name"],
            payload["District_Name"],
            payload["Crop_Year"],
            payload["Season"],
            payload["Crop"],
            payload["Area"],
            payload["Prediction"],
            datetime.utcnow().isoformat(timespec="seconds")
        ))
        conn.commit()

# ========= Routes =========

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # ✅ Admin credentials → Dashboard
    if username == "admin" and password == "admin":
        return redirect('/dashboard')

    # ✅ Salim credentials → Prediction page
    elif username == "salim" and password == "salim":
        return redirect('/predict')

    else:
        return "❌ Invalid credentials"


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/face_login', methods=['POST'])
def face_login():
    global known_faces
    if known_faces is None:
        known_faces = load_known_faces()

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        faces = DeepFace.extract_faces(
            img_path=frame, detector_backend='opencv', enforce_detection=False
        )
        if len(faces) == 0:
            return jsonify({"status": "failed"})

        for face in faces:
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            face_crop = frame[y:y+h, x:x+w]

            embedding = DeepFace.represent(
                img_path=face_crop, model_name="VGG-Face", enforce_detection=False
            )[0]["embedding"]

            for name, ref_embedding in known_faces.items():
                dist = cosine(ref_embedding, embedding)
                if dist < threshold:
                    session['user'] = name
                    return jsonify({"status": "success", "user": name})

        return jsonify({"status": "failed_attempt"})

    except Exception as e:
        print("⚠️ Face detection error:", e)
        return jsonify({"status": "failed"})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            import pandas as pd
            user = session.get('user', 'guest')

            state = request.form["State_Name"]
            district = request.form["District_Name"]
            year = int(request.form["Crop_Year"])
            season = request.form["Season"]
            crop = request.form["Crop"]
            area = float(request.form["Area"])

            input_df = pd.DataFrame([{
                "State_Name": state,
                "District_Name": district,
                "Crop_Year": year,
                "Season": season,
                "Crop": crop,
                "Area": area
            }])

            prediction = model.predict(input_df)[0]
            prediction = round(float(prediction), 2)

            # ✅ Log to SQLite
            log_prediction(user, {
                "State_Name": state,
                "District_Name": district,
                "Crop_Year": year,
                "Season": season,
                "Crop": crop,
                "Area": area,
                "Prediction": prediction
            })

            # Render same page (AJAX will scrape #prediction-value)
            return render_template("form.html", prediction=prediction)

        except Exception as e:
            import traceback
            print("❌ Prediction error:", e)
            traceback.print_exc()
            return render_template("form.html", prediction="⚠️ Error")
    return render_template("form.html")

@app.route("/dashboard")
def dashboard():
    # Pull recent logs and aggregates
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Recent predictions
        cur.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()

        # Aggregations
        cur.execute("""
            SELECT crop, COUNT(*) as cnt
            FROM logs
            GROUP BY crop
            ORDER BY cnt DESC
            LIMIT 10
        """)
        top_crops = cur.fetchall()

        cur.execute("""
            SELECT season, COUNT(*) as cnt
            FROM logs
            GROUP BY season
            ORDER BY cnt DESC
        """)
        by_season = cur.fetchall()

        cur.execute("""
            SELECT substr(created_at, 1, 10) as day, COUNT(*) as cnt
            FROM logs
            GROUP BY day
            ORDER BY day ASC
            LIMIT 30
        """)
        by_day = cur.fetchall()

    # Convert to plain lists for Chart.js
    top_crop_labels = [r["crop"] for r in top_crops]
    top_crop_values = [r["cnt"] for r in top_crops]

    season_labels = [r["season"] for r in by_season]
    season_values = [r["cnt"] for r in by_season]

    day_labels = [r["day"] for r in by_day]
    day_values = [r["cnt"] for r in by_day]

    return render_template(
        "dashboard.html",
        rows=rows,
        top_crop_labels=top_crop_labels,
        top_crop_values=top_crop_values,
        season_labels=season_labels,
        season_values=season_values,
        day_labels=day_labels,
        day_values=day_values,
        current_user=session.get('user', 'guest')
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


