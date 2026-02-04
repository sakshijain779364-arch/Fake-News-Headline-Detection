from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import hashlib
import joblib
import os

# Get the directory of the current script
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))
app.secret_key = 'supersecretkey'  # Change this in production

# User storage file
USER_FILE = os.path.join(basedir, 'users.txt')

def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                username, pw_hash = line.strip().split(',')
                users[username] = pw_hash
    return users

def save_user(username, password):
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    with open(USER_FILE, 'a', encoding='utf-8') as f:
        f.write(f'{username},{pw_hash}\n')

def check_user(username, password):
    users = load_users()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    return users.get(username) == pw_hash
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_user(username, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            error = 'Username already exists.'
        else:
            save_user(username, password)
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Load saved model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

@app.route("/")
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", username=session['username'])

@app.route("/predict", methods=["POST"])
def predict():
    # This is the correct predict function to keep
    data = request.get_json()
    headline = data.get("headline", "").strip()
    body_text = data.get("body_text", "").strip()
    
    if not headline:
        return jsonify({"error": "Please provide a headline."}), 400

    # Combine headline and body_text for prediction
    combined_text = headline + " " + body_text if body_text else headline
    
    X = vectorizer.transform([combined_text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]  # 0 = Real, 1 = Fake

    result_label = "Real" if pred == 0 else "Fake"
    confidence = float(max(proba))  # e.g., 0.86

    return jsonify({
        "label": result_label,
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

    
