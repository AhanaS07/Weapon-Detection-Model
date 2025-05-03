from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
from weapon_detection_system import weapon_detection_system
from detection_api import detection_api
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key in production
app.register_blueprint(detection_api)

task_status = {"completed": False}

def run_weapon_detection():
    global task_status
    task_status["completed"] = False
    for processed_image in weapon_detection_system():
        task_status["completed"] = False
        yield f"data:{processed_image}\n\n"
    task_status["completed"] = True

@app.route('/task_status')
def task_status_check():
    return jsonify(task_status)

@app.route('/stream')
def stream():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    return Response(run_weapon_detection(), mimetype="text/event-stream")

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('loading'))
        else:
            error = "Invalid username or password."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/loading')
def loading():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    return render_template('loading.html')

@app.route('/detect-weapons')
def detect_weapons_page():
    """Renders the weapon detection page for manual image uploads."""
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    return render_template('detect.html')

@app.route('/detection-results/<filename>')
def detection_results(filename):
    """Display detection results for a specific image."""
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    return render_template('results.html', image_filename=filename)

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

