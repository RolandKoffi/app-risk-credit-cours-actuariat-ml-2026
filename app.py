from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'risk_credit_secret'
MODEL_NAME = "modele_arbre_risk_cred.pkl"

USER_DB = 'users.xlsx'
if not os.path.exists(USER_DB):
    df = pd.DataFrame(columns=['prenom', 'nom', 'username', 'password'])
    df.to_excel(USER_DB, index=False)

model = joblib.load(os.path.join(f"./saved_models/{MODEL_NAME}"))

FEATURE_ORDER = [
    'AGE', 'SALAIRE_MENSUEL', 'DUREE_EMPLOI', 'MONTANT_CREDIT', 'DUREE_CREDIT',
    'NB_CREDITS_EN_COURS', 'RATIO_ENDETTEMENT', 'NB_INCIDENTS',
    'MENSUALITE_ESTIMEE', 'RATIO_CREDIT_SALAIRE', 'A_INCIDENTS', 'ANCIENNETE_ANNEES',
    'HISTORIQUE_ENC', 'TYPE_EMPLOI_Indépendant', 'TYPE_EMPLOI_Salarié',
    'TYPE_EMPLOI_Sans emploi', 'GARANTIE_Caution', 'GARANTIE_Immobilière', 'GARANTIE_Véhicule'
]


@app.route('/')
def home():
    if 'prenom' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', prenom=session['prenom'], prediction=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = pd.read_excel(USER_DB)
        user = users[users['username'] == username]
        if not user.empty and check_password_hash(user.iloc[0]['password'], password):
            session['prenom'] = user.iloc[0]['prenom']
            return redirect(url_for('home'))
        return render_template('login.html', error='Identifiants incorrects.')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        prenom = request.form.get('prenom')
        nom = request.form.get('nom')
        username = request.form.get('username')
        password = generate_password_hash(request.form.get('password'))

        try:
            users = pd.read_excel(USER_DB)
            if username in users['username'].values:
                return render_template('register.html', register_error="Nom d'utilisateur déjà utilisé.")
            new_user = {'prenom': prenom, 'nom': nom, 'username': username, 'password': password}
            users = pd.concat([users, pd.DataFrame([new_user])], ignore_index=True)
            users.to_excel(USER_DB, index=False)
            return render_template('login.html', register_success="Inscription réussie. Connectez-vous.")
        except Exception as e:
            print("Erreur inscription :", e)
            return render_template('register.html', register_error="Une erreur est survenue.")

    return render_template('register.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'prenom' not in session:
        return redirect(url_for('login'))

    try:
        f = request.form

        type_emploi = f.get('TYPE_EMPLOI', '')
        garantie = f.get('GARANTIE', '')

        data = {
            'AGE': float(f['AGE']),
            'SALAIRE_MENSUEL': float(f['SALAIRE_MENSUEL']),
            'DUREE_EMPLOI': float(f['DUREE_EMPLOI']),
            'MONTANT_CREDIT': float(f['MONTANT_CREDIT']),
            'DUREE_CREDIT': float(f['DUREE_CREDIT']),
            'NB_CREDITS_EN_COURS': float(f['NB_CREDITS_EN_COURS']),
            'RATIO_ENDETTEMENT': float(f['RATIO_ENDETTEMENT']),
            'NB_INCIDENTS': float(f['NB_INCIDENTS']),
            'MENSUALITE_ESTIMEE': float(f['MENSUALITE_ESTIMEE']),
            'RATIO_CREDIT_SALAIRE': float(f['RATIO_CREDIT_SALAIRE']),
            'A_INCIDENTS': float(f['A_INCIDENTS']),
            'ANCIENNETE_ANNEES': float(f['ANCIENNETE_ANNEES']),
            'HISTORIQUE_ENC': float(f['HISTORIQUE_ENC']),
            'TYPE_EMPLOI_Indépendant': 1.0 if type_emploi == 'Indépendant' else 0.0,
            'TYPE_EMPLOI_Salarié': 1.0 if type_emploi == 'Salarié' else 0.0,
            'TYPE_EMPLOI_Sans emploi': 1.0 if type_emploi == 'Sans emploi' else 0.0,
            'GARANTIE_Caution': 1.0 if garantie == 'Caution' else 0.0,
            'GARANTIE_Immobilière': 1.0 if garantie == 'Immobilière' else 0.0,
            'GARANTIE_Véhicule': 1.0 if garantie == 'Véhicule' else 0.0,
        }

        X = pd.DataFrame([data])[FEATURE_ORDER]
        prediction = int(model.predict(X)[0])

        return render_template('home.html', prenom=session['prenom'], prediction=prediction)

    except Exception as e:
        print("Erreur prédiction :", e)
        return render_template('home.html', prenom=session['prenom'], prediction=None, error="Erreur lors de la prédiction.")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
