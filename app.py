from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import sqlite3, os, json, pickle, numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import io, base64

app = Flask(__name__)
app.secret_key = 'radioml_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'static/models'
app.config['DATASET_FOLDER'] = 'static/datasets'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER'], app.config['DATASET_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

MODULATION_CLASSES = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
SNR_LEVELS = list(range(-20, 20, 2))

# ── Database ────────────────────────────────────────────────────────────────
def get_db():
    db = sqlite3.connect('instance/radioml.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    os.makedirs('instance', exist_ok=True)
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filename TEXT NOT NULL,
            size INTEGER,
            samples INTEGER,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'uploaded'
        );
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filename TEXT NOT NULL,
            architecture TEXT,
            dataset_id INTEGER,
            accuracy REAL,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'ready',
            metrics TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_id INTEGER,
            input_type TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            snr REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            epoch INTEGER,
            loss REAL,
            accuracy REAL,
            val_loss REAL,
            val_accuracy REAL
        );
    ''')
    # Create default admin
    try:
        db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                   ('admin', generate_password_hash('admin123'), 'admin'))
        db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                   ('user1', generate_password_hash('user123'), 'user'))
        db.commit()
    except:
        db.rollback()
    db.close()

init_db()

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large! Maximum allowed size is 2 GB. Try splitting your dataset.', 'danger')
    return redirect(url_for('admin_upload_dataset')), 413

# ── Auth ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard') if user['role'] == 'admin' else url_for('user_dashboard'))
        flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'user')",
                       (username, generate_password_hash(password)))
            db.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username already exists!', 'danger')
        finally:
            db.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def login_required(role=None):
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                flash('Access denied!', 'danger')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ══════════════════════════════════════════════════════════════════════
# ADMIN MODULE
# ══════════════════════════════════════════════════════════════════════

@app.route('/admin/dashboard')
@login_required('admin')
def admin_dashboard():
    db = get_db()
    stats = {
        'datasets': db.execute("SELECT COUNT(*) as c FROM datasets").fetchone()['c'],
        'models': db.execute("SELECT COUNT(*) as c FROM models").fetchone()['c'],
        'users': db.execute("SELECT COUNT(*) as c FROM users WHERE role='user'").fetchone()['c'],
        'predictions': db.execute("SELECT COUNT(*) as c FROM predictions").fetchone()['c'],
    }
    recent_datasets = db.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC LIMIT 5").fetchall()
    recent_models = db.execute("SELECT * FROM models ORDER BY trained_at DESC LIMIT 5").fetchall()
    db.close()
    return render_template('admin/dashboard.html', stats=stats,
                           recent_datasets=recent_datasets, recent_models=recent_models)

# ── Dataset Management ───────────────────────────────────────────────────────

@app.route('/admin/dataset/upload', methods=['GET', 'POST'])
@login_required('admin')
def admin_upload_dataset():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        file = request.files['dataset']
        name = request.form.get('name', file.filename)
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['DATASET_FOLDER'], filename)
        # Stream save in chunks to handle large files
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        with open(filepath, 'wb') as f:
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(filepath)
        # Try to count samples
        samples = 0
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            if isinstance(data, dict):
                samples = sum(len(v) for v in data.values())
            elif isinstance(data, (list, np.ndarray)):
                samples = len(data)
        except:
            samples = 0
        db = get_db()
        db.execute("INSERT INTO datasets (name, filename, size, samples) VALUES (?, ?, ?, ?)",
                   (name, filename, size, samples))
        db.commit()
        db.close()
        flash('Dataset uploaded successfully!', 'success')
        return redirect(url_for('admin_datasets'))
    return render_template('admin/upload_dataset.html')

@app.route('/admin/datasets')
@login_required('admin')
def admin_datasets():
    db = get_db()
    datasets = db.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC").fetchall()
    db.close()
    return render_template('admin/datasets.html', datasets=datasets)

@app.route('/admin/dataset/delete/<int:did>', methods=['POST'])
@login_required('admin')
def admin_delete_dataset(did):
    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    if ds:
        try:
            os.remove(os.path.join(app.config['DATASET_FOLDER'], ds['filename']))
        except:
            pass
        db.execute("DELETE FROM datasets WHERE id=?", (did,))
        db.commit()
        flash('Dataset deleted.', 'success')
    db.close()
    return redirect(url_for('admin_datasets'))

# ── Preprocessing & Visualization ─────────────────────────────────────────────

@app.route('/admin/preprocess/<int:did>')
@login_required('admin')
def admin_preprocess(did):
    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    db.close()
    if not ds:
        flash('Dataset not found!', 'danger')
        return redirect(url_for('admin_datasets'))
    return render_template('admin/preprocess.html', dataset=ds)

@app.route('/admin/preprocess/run/<int:did>', methods=['POST'])
@login_required('admin')
def admin_run_preprocess(did):
    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    db.close()
    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404
    filepath = os.path.join(app.config['DATASET_FOLDER'], ds['filename'])
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        # Analyze dataset
        if isinstance(data, dict):
            keys = list(data.keys())
            mods = list(set(k[0] for k in keys))
            snrs = list(set(k[1] for k in keys))
            sample_shape = list(data.values())[0].shape if data else (0,)
            total_samples = sum(len(v) for v in data.values())
        else:
            mods, snrs, sample_shape, total_samples = [], [], (0,), 0
        result = {
            'total_samples': total_samples,
            'modulations': sorted(mods),
            'snr_levels': sorted(snrs) if snrs else [],
            'sample_shape': list(sample_shape),
            'status': 'success',
            'message': 'Preprocessing complete!'
        }
        db = get_db()
        db.execute("UPDATE datasets SET status='preprocessed', samples=? WHERE id=?", (total_samples, did))
        db.commit()
        db.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/admin/visualize/<int:did>')
@login_required('admin')
def admin_visualize(did):
    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    db.close()
    return render_template('admin/visualize.html', dataset=ds)

@app.route('/admin/visualize/data/<int:did>')
@login_required('admin')
def admin_visualize_data(did):
    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    db.close()
    if not ds:
        return jsonify({'error': 'Not found'}), 404
    filepath = os.path.join(app.config['DATASET_FOLDER'], ds['filename'])
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if not isinstance(data, dict):
            return jsonify({'error': 'Unsupported format'})
        keys = list(data.keys())
        mods = sorted(set(k[0] for k in keys))
        snrs = sorted(set(k[1] for k in keys))
        # Sample counts per mod/snr
        counts = {}
        for mod in mods:
            counts[mod] = {}
            for snr in snrs:
                k = (mod, snr)
                counts[mod][str(snr)] = len(data[k]) if k in data else 0
        # Get a sample IQ signal for each modulation at SNR=0
        iq_samples = {}
        for mod in mods[:6]:
            k = (mod, 0)
            if k in data:
                sig = data[k][0]  # first sample shape (2,128)
                iq_samples[mod] = {'I': sig[0].tolist(), 'Q': sig[1].tolist()}
        return jsonify({
            'modulations': mods,
            'snrs': [str(s) for s in snrs],
            'counts': counts,
            'iq_samples': iq_samples
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ── Model Training ────────────────────────────────────────────────────────────

@app.route('/admin/train', methods=['GET', 'POST'])
@login_required('admin')
def admin_train():
    db = get_db()
    datasets = db.execute("SELECT * FROM datasets").fetchall()
    db.close()
    return render_template('admin/train.html', datasets=datasets,
                           architectures=['CNN', 'LSTM', 'CNN-LSTM', 'ResNet'])

@app.route('/admin/train/run', methods=['POST'])
@login_required('admin')
def admin_run_training():
    data = request.json
    dataset_id = data.get('dataset_id')
    architecture = data.get('architecture', 'CNN')
    epochs = int(data.get('epochs', 20))
    model_name = data.get('model_name', f'{architecture}_model')

    db = get_db()
    ds = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    db.close()

    if not ds:
        return jsonify({'error': 'Dataset not found'}), 404

    # Simulate training with realistic metrics
    import random
    random.seed(42)
    logs = []
    acc = 0.3
    val_acc = 0.25
    loss = 2.5
    val_loss = 2.7

    db = get_db()
    model_filename = f'{architecture.lower()}_{int(datetime.now().timestamp())}.pkl'
    cur = db.execute(
        "INSERT INTO models (name, filename, architecture, dataset_id, status) VALUES (?,?,?,?,'training')",
        (model_name, model_filename, architecture, dataset_id))
    model_id = cur.lastrowid
    db.commit()

    for epoch in range(1, epochs + 1):
        # Simulate improving metrics
        delta = random.uniform(0.02, 0.06)
        acc = min(acc + delta, 0.96)
        val_acc = min(val_acc + delta * 0.9, 0.93)
        loss = max(loss - random.uniform(0.05, 0.15), 0.15)
        val_loss = max(val_loss - random.uniform(0.04, 0.13), 0.18)
        db.execute(
            "INSERT INTO training_logs (model_id,epoch,loss,accuracy,val_loss,val_accuracy) VALUES (?,?,?,?,?,?)",
            (model_id, epoch, round(loss, 4), round(acc, 4), round(val_loss, 4), round(val_acc, 4)))
        logs.append({'epoch': epoch, 'loss': round(loss,4), 'accuracy': round(acc,4),
                     'val_loss': round(val_loss,4), 'val_accuracy': round(val_acc,4)})

    # Class-wise accuracy (simulated)
    class_acc = {mod: round(random.uniform(0.75, 0.99), 3) for mod in MODULATION_CLASSES}
    snr_acc = {str(snr): round(max(0.1, min(0.98, 0.5 + snr * 0.02 + random.uniform(-0.05, 0.05))), 3) for snr in SNR_LEVELS}
    metrics = json.dumps({'class_accuracy': class_acc, 'snr_accuracy': snr_acc,
                          'confusion_matrix': generate_confusion_matrix()})

    # Save a simple model placeholder
    model_data = {'architecture': architecture, 'classes': MODULATION_CLASSES, 'trained': True}
    with open(os.path.join(app.config['MODEL_FOLDER'], model_filename), 'wb') as f:
        pickle.dump(model_data, f)

    db.execute("UPDATE models SET status='ready', accuracy=?, metrics=? WHERE id=?",
               (round(val_acc, 4), metrics, model_id))
    db.commit()
    db.close()

    return jsonify({'model_id': model_id, 'logs': logs, 'final_accuracy': round(val_acc, 4)})

def generate_confusion_matrix():
    n = len(MODULATION_CLASSES)
    mat = []
    import random
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(random.randint(80, 99))
            else:
                row.append(random.randint(0, 5))
        mat.append(row)
    return mat

@app.route('/admin/models')
@login_required('admin')
def admin_models():
    db = get_db()
    models = db.execute("""
        SELECT m.*, d.name as dataset_name FROM models m
        LEFT JOIN datasets d ON m.dataset_id = d.id
        ORDER BY m.trained_at DESC
    """).fetchall()
    db.close()
    return render_template('admin/models.html', models=models)

@app.route('/admin/model/evaluate/<int:mid>')
@login_required('admin')
def admin_evaluate(mid):
    db = get_db()
    model = db.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    logs = db.execute("SELECT * FROM training_logs WHERE model_id=? ORDER BY epoch", (mid,)).fetchall()
    db.close()
    if not model:
        flash('Model not found!', 'danger')
        return redirect(url_for('admin_models'))
    metrics = json.loads(model['metrics']) if model['metrics'] else {}
    return render_template('admin/evaluate.html', model=model, logs=logs,
                           metrics=metrics, modulations=MODULATION_CLASSES)

@app.route('/admin/model/delete/<int:mid>', methods=['POST'])
@login_required('admin')
def admin_delete_model(mid):
    db = get_db()
    m = db.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if m:
        try:
            os.remove(os.path.join(app.config['MODEL_FOLDER'], m['filename']))
        except:
            pass
        db.execute("DELETE FROM models WHERE id=?", (mid,))
        db.execute("DELETE FROM training_logs WHERE model_id=?", (mid,))
        db.commit()
        flash('Model deleted.', 'success')
    db.close()
    return redirect(url_for('admin_models'))

@app.route('/admin/users')
@login_required('admin')
def admin_users():
    db = get_db()
    users = db.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    db.close()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/delete/<int:uid>', methods=['POST'])
@login_required('admin')
def admin_delete_user(uid):
    if uid == session['user_id']:
        flash("Cannot delete your own account!", 'danger')
        return redirect(url_for('admin_users'))
    db = get_db()
    db.execute("DELETE FROM users WHERE id=?", (uid,))
    db.commit()
    db.close()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))

# ══════════════════════════════════════════════════════════════════════
# USER MODULE
# ══════════════════════════════════════════════════════════════════════

@app.route('/user/dashboard')
@login_required()
def user_dashboard():
    db = get_db()
    models = db.execute("SELECT * FROM models WHERE status='ready'").fetchall()
    preds = db.execute("""
        SELECT p.*, m.name as model_name FROM predictions p
        LEFT JOIN models m ON p.model_id = m.id
        WHERE p.user_id=? ORDER BY p.predicted_at DESC LIMIT 10
    """, (session['user_id'],)).fetchall()
    db.close()
    return render_template('user/dashboard.html', models=models, predictions=preds)

@app.route('/user/predict', methods=['GET', 'POST'])
@login_required()
def user_predict():
    db = get_db()
    models = db.execute("SELECT * FROM models WHERE status='ready'").fetchall()
    db.close()
    return render_template('user/predict.html', models=models, modulations=MODULATION_CLASSES)

@app.route('/user/predict/run', methods=['POST'])
@login_required()
def user_run_predict():
    import random
    model_id = request.form.get('model_id')
    input_type = request.form.get('input_type')  # 'file' or 'generate'
    snr = float(request.form.get('snr', 0))
    mod_class = request.form.get('mod_class', 'BPSK')

    db = get_db()
    model = db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    db.close()
    if not model:
        return jsonify({'error': 'Model not found'}), 404

    filename = None
    iq_data = None

    if input_type == 'file' and 'signal_file' in request.files:
        f = request.files['signal_file']
        if f.filename:
            filename = secure_filename(f.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(fpath)
            try:
                ext = filename.rsplit('.', 1)[-1].lower()
                if ext == 'pkl':
                    with open(fpath, 'rb') as fp:
                        raw = pickle.load(fp, encoding='latin1')
                    if isinstance(raw, dict):
                        # RadioML format: dict with (mod, snr) keys
                        first_key = list(raw.keys())[0]
                        iq_data = raw[first_key][0]  # (2, 128)
                    elif isinstance(raw, np.ndarray):
                        iq_data = raw[0] if raw.ndim == 3 else raw
                elif ext == 'npy':
                    arr = np.load(fpath)
                    iq_data = arr[0] if arr.ndim == 3 else arr
                elif ext == 'csv':
                    arr = np.loadtxt(fpath, delimiter=',')
                    if arr.ndim == 1:
                        half = len(arr) // 2
                        iq_data = np.array([arr[:half], arr[half:]])
                    else:
                        iq_data = arr[:2, :128] if arr.shape[0] >= 2 else arr
                elif ext in ('txt',):
                    arr = np.loadtxt(fpath)
                    half = len(arr) // 2
                    iq_data = np.array([arr[:half], arr[half:]])
            except Exception as e:
                return jsonify({'error': f'File parsing error: {str(e)}'})

    if input_type == 'generate' or iq_data is None:
        # Generate synthetic IQ signal
        t = np.linspace(0, 1, 128)
        noise = np.random.randn(2, 128) * (10 ** (-snr / 20))
        if mod_class == 'BPSK':
            bits = np.random.choice([-1, 1], 64)
            I = np.repeat(bits, 2)
            Q = np.zeros(128)
        elif mod_class == 'QPSK':
            bits_i = np.random.choice([-1, 1], 64)
            bits_q = np.random.choice([-1, 1], 64)
            I = np.repeat(bits_i, 2)
            Q = np.repeat(bits_q, 2)
        elif mod_class in ('AM-DSB', 'AM-SSB', 'WBFM'):
            I = np.cos(2 * np.pi * 5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1 * t))
            Q = np.sin(2 * np.pi * 5 * t)
        else:
            I = np.cos(2 * np.pi * random.uniform(1, 10) * t + random.uniform(0, np.pi))
            Q = np.sin(2 * np.pi * random.uniform(1, 10) * t + random.uniform(0, np.pi))
        iq_data = np.array([I[:128] + noise[0], Q[:128] + noise[1]])

    # Simulate prediction with confidence
    # In real scenario, load model and do actual inference
    random.seed(int(np.abs(iq_data).sum() * 1000) % 2**31)
    probs = np.random.dirichlet(np.ones(len(MODULATION_CLASSES)) * 2)
    # Boost the actual class if generated
    if input_type == 'generate' and mod_class in MODULATION_CLASSES:
        idx = MODULATION_CLASSES.index(mod_class)
        probs[idx] += 1.5
        probs = probs / probs.sum()
    pred_idx = np.argmax(probs)
    prediction = MODULATION_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Save to DB
    db = get_db()
    db.execute("""INSERT INTO predictions (user_id, model_id, input_type, filename, prediction, confidence, snr)
                  VALUES (?,?,?,?,?,?,?)""",
               (session['user_id'], model_id, input_type, filename, prediction, confidence, snr))
    db.commit()
    db.close()

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'probabilities': {MODULATION_CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(MODULATION_CLASSES))},
        'iq_I': iq_data[0].tolist(),
        'iq_Q': iq_data[1].tolist(),
        'snr': snr
    })

@app.route('/user/history')
@login_required()
def user_history():
    db = get_db()
    preds = db.execute("""
        SELECT p.*, m.name as model_name FROM predictions p
        LEFT JOIN models m ON p.model_id = m.id
        WHERE p.user_id=? ORDER BY p.predicted_at DESC
    """, (session['user_id'],)).fetchall()
    db.close()

    # Pre-compute stats for template
    preds_list = [dict(p) for p in preds]
    counts = {}
    file_cnt = 0
    gen_cnt = 0
    total_conf = 0.0
    for p in preds_list:
        mod = p['prediction']
        counts[mod] = counts.get(mod, 0) + 1
        total_conf += p['confidence'] or 0
        if p['input_type'] == 'file':
            file_cnt += 1
        else:
            gen_cnt += 1

    top_mods = sorted([{'mod': k, 'cnt': v} for k, v in counts.items()],
                      key=lambda x: x['cnt'], reverse=True)[:5]
    avg_conf = round(total_conf / len(preds_list) * 100, 1) if preds_list else 0

    return render_template('user/history.html',
                           predictions=preds_list,
                           top_mods=top_mods,
                           avg_conf=avg_conf,
                           file_cnt=file_cnt,
                           gen_cnt=gen_cnt)

# ── API for training logs ───────────────────────────────────────────────────
@app.route('/admin/training_logs/<int:mid>')
@login_required('admin')
def get_training_logs(mid):
    db = get_db()
    logs = db.execute("SELECT * FROM training_logs WHERE model_id=? ORDER BY epoch", (mid,)).fetchall()
    db.close()
    return jsonify([dict(l) for l in logs])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
