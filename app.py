
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sne5700'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load ML model and scaler
model = joblib.load('refined_best_svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------- DATABASE MODELS -------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    specialization = db.Column(db.String(100))
    hospital = db.Column(db.String(100))
    contact = db.Column(db.String(20))
    password = db.Column(db.String(200), nullable=False)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    chestpain = db.Column(db.Integer, nullable=False)
    restingBP = db.Column(db.Integer, nullable=False)
    serumcholestrol = db.Column(db.Integer, nullable=False)
    fastingbloodsugar = db.Column(db.Integer, nullable=False)
    restingrelectro = db.Column(db.Integer, nullable=False)
    maxheartrat = db.Column(db.Integer, nullable=False)
    exerciseangia = db.Column(db.Integer, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    slope = db.Column(db.Integer, nullable=False)
    noofmajor = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50))

    doctor = db.relationship('User', backref=db.backref('patients', lazy=True))

# ------------------- LOGIN MANAGER -------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------- ROUTES -------------------
@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('homes.html', username=current_user.name)
    else:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        specialization = request.form['specialization']
        hospital = request.form['hospital']
        contact = request.form['contact']

        if User.query.filter_by(email=email).first():
            flash('Email already exists!')
            return redirect(url_for('register'))

        new_user = User(name=name, email=email, password=password,
                        specialization=specialization, hospital=hospital, contact=contact)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(name=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/patient')
@login_required
def patient():
    return render_template('patient.html')

@app.route('/homes')
@login_required
def homes():
    return render_template('homes.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/enter_patient', methods=['GET', 'POST'])
@login_required
def enter_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        chestpain = int(request.form['chestpain'])
        restingBP = int(request.form['restingBP'])
        serumcholestrol = int(request.form['serumcholestrol']) 
        fastingbloodsugar = int(request.form['fastingbloodsugar'])
        restingrelectro = int(request.form['restingrelectro'])
        maxheartrat = int(request.form['maxheartrat'])
        exerciseangia = int(request.form['exerciseangia'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        noofmajor = int(request.form['noofmajor'])

        input_data = [
            age, gender, chestpain, restingBP, serumcholestrol,
            fastingbloodsugar, restingrelectro, maxheartrat,
            exerciseangia, oldpeak, slope, noofmajor
        ]

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        prediction_result = "Has Heart Disease" if prediction == 1 else "No Heart Disease"

        new_patient = Patient(
            doctor_id=current_user.id,
            name=name,
            age=age,
            gender=gender,
            chestpain=chestpain,
            restingBP=restingBP,
            serumcholestrol=serumcholestrol,
            fastingbloodsugar=fastingbloodsugar,
            restingrelectro=restingrelectro,
            maxheartrat=maxheartrat,
            exerciseangia=exerciseangia,
            oldpeak=oldpeak,
            slope=slope,
            noofmajor=noofmajor,
            prediction=prediction_result
        )
        db.session.add(new_patient)
        db.session.commit()
        flash('Patient details saved and prediction done.')
        return redirect(url_for('view_patient_details'))

    return render_template('enter_patient.html')

@app.route('/view_patients')
@login_required
def view_patient_details():
    patients = Patient.query.filter_by(doctor_id=current_user.id).all()
    return render_template('view_patients.html', patients=patients)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        name = request.form['name']
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        chestpain = int(request.form['chestpain'])
        restingBP = int(request.form['restingBP'])
        serumcholestrol = int(request.form['serumcholestrol'])
        fastingbloodsugar = int(request.form['fastingbloodsugar'])
        restingrelectro = int(request.form['restingrelectro'])
        maxheartrat = int(request.form['maxheartrat'])
        exerciseangia = int(request.form['exerciseangia'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        noofmajor = int(request.form['noofmajor'])

        input_data = [
            age, gender, chestpain, restingBP, serumcholestrol,
            fastingbloodsugar, restingrelectro, maxheartrat,
            exerciseangia, oldpeak, slope, noofmajor
        ]

        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        prediction_result = "Has Heart Disease" if prediction == 1 else "No Heart Disease"

        new_patient = Patient(
            doctor_id=current_user.id,
            name=name,
            age=age,
            gender=gender,
            chestpain=chestpain,
            restingBP=restingBP,
            serumcholestrol=serumcholestrol,
            fastingbloodsugar=fastingbloodsugar,
            restingrelectro=restingrelectro,
            maxheartrat=maxheartrat,
            exerciseangia=exerciseangia,
            oldpeak=oldpeak,
            slope=slope,
            noofmajor=noofmajor,
            prediction=prediction_result
        )
        db.session.add(new_patient)
        db.session.commit()

        return render_template("predict.html",
                                name=name,
                                age=age,
                                gender=gender,
                                chestpain=chestpain,
                                restingBP=restingBP,
                                serumcholestrol=serumcholestrol,
                                fastingbloodsugar=fastingbloodsugar,
                                restingrelectro=restingrelectro,
                                maxheartrat=maxheartrat,
                                exerciseangia=exerciseangia,
                                oldpeak=oldpeak,
                                slope=slope,
                                noofmajor=noofmajor,
                                prediction=prediction_result)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        return "Unauthorized", 403
    db.session.delete(patient)
    db.session.commit()
    return redirect(url_for('view_patient_details'))

# ------------------- MAIN -------------------
if __name__ == '__main__':
    if os.path.exists("database.db"):
        os.remove("database.db")

    with app.app_context():
        db.create_all()

    app.run(debug=True)
