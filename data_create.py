import random
import math
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

def generate_synthetic_loan_data(n_rows=5000, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    rows = []
    db_records = []

    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
    educations = ["High School", "Bachelor", "Master", "PhD", "Other"]
    employment_statuses = ["Employed", "Self-employed", "Unemployed", "Student", "Retired"]
    purposes = ["Personal", "Home", "Car", "Education", "Business", "Other"]

    first_names_m = ["Александр", "Дмитрий", "Максим", "Иван", "Сергей", "Андрей", "Алексей", "Михаил"]
    first_names_f = ["Мария", "Анна", "Елена", "Ольга", "Татьяна", "Наталья", "Екатерина", "София"]
    last_names_roots = ["Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", "Соколов"]

    for _ in range(n_rows):
        gender = random.choice(genders)
        root_last = random.choice(last_names_roots)
        if gender == "Male":
            fname = random.choice(first_names_m)
            lname = root_last
        else:
            fname = random.choice(first_names_f)
            lname = root_last + "а"
        full_name = f"{lname} {fname}"
        
        age = int(np.random.randint(18, 70))
        birth_year = 2025 - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        birth_date = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

        marital = random.choice(marital_statuses)
        education = random.choices(educations, weights=[0.3, 0.35, 0.2, 0.05, 0.1])[0] 
        employment = random.choices(employment_statuses, weights=[0.6, 0.15, 0.1, 0.1, 0.05])[0]
        purpose = random.choice(purposes)

        annual_income = float(np.exp(np.random.uniform(np.log(50_000), np.log(5_000_000))))

        loan_income_ratio = np.random.uniform(0.1, 5.0)
        loan_amount = float(annual_income * loan_income_ratio)

        base_score = np.random.uniform(300, 850) 
        if employment == "Unemployed": base_score -= 50
        if late_payments := np.random.poisson(0.5): base_score -= (late_payments * 30)
        credit_score = int(np.clip(base_score, 300, 850))
        
        existing_loans = int(np.random.randint(0, 10))
        if credit_score < 500:
            late_payments_last_year = int(np.random.randint(0, 10))
        else:
            late_payments_last_year = int(np.random.poisson(0.5))

        score_val = 0.0
        score_val += 2.5 * math.log10(annual_income) 
        ratio = loan_amount / annual_income
        if ratio > 3.0: score_val -= 4.0
        elif ratio > 1.0: score_val -= 2.0
        elif ratio < 0.5: score_val += 1.5
        
        if education in ["PhD", "Master"]: score_val += 1.2
        elif education == "Bachelor": score_val += 0.7
        elif education == "High School": score_val -= 0.5
        else: score_val -= 0.8

        norm_score = (credit_score - 300) / 550.0
        score_val += norm_score * 4.0
        score_val -= late_payments_last_year * 1.0
        if marital == "Married": score_val += 0.5
        elif marital in ["Single", "Divorced"]: score_val -= 0.3

        bias = -14.5
        noise = np.random.normal(0, 0.5)
        final_logit = score_val + bias + noise
        p_approve = 1 / (1 + math.exp(-final_logit)) 
        approved_flag = 1 if random.random() < p_approve else 0

        rows.append({
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital,
            "EducationLevel": education,
            "EmploymentStatus": employment,
            "AnnualIncome": round(annual_income, 2),
            "LoanAmountRequested": round(loan_amount, 2),
            "PurposeOfLoan": purpose,
            "CreditScore": credit_score,
            "ExistingLoansCount": existing_loans,
            "LatePaymentsLastYear": late_payments_last_year,
            "LoanApproved": approved_flag 
        })
        db_records.append((full_name, birth_date, credit_score))

    df = pd.DataFrame(rows)
    return df, db_records

def create_and_fill_db(db_records):
    db_path = "data/credit_bureau.db"
    if os.path.exists(db_path): os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_scores (
            full_name TEXT,
            birth_date TEXT,
            credit_score REAL,
            PRIMARY KEY (full_name, birth_date)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS application_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            full_name TEXT,
            birth_date TEXT,
            
            age INTEGER,
            gender TEXT,
            marital_status TEXT,
            education TEXT,
            employment TEXT,
            income REAL,
            loan_amount REAL,
            purpose TEXT,
            existing_loans INTEGER,
            late_payments INTEGER,
            
            model_used TEXT,
            predicted_probability REAL,
            bureau_score_used REAL
        )
    """)
    
    cur.executemany("INSERT OR IGNORE INTO credit_scores (full_name, birth_date, credit_score) VALUES (?, ?, ?)", db_records)
    conn.commit()
    print(f"БД инициализирована")
    conn.close()


if __name__ == "__main__":
    df, db_records = generate_synthetic_loan_data(n_rows=5000, random_state=42)
    df.to_csv("data/synthetic_loan_approval.csv", index=False)
    create_and_fill_db(db_records)
    print("Данные сгенерированы")
    conn = sqlite3.connect("data/credit_bureau.db")
    cur = conn.cursor()
    cur.execute("SELECT full_name, birth_date, credit_score FROM credit_scores ORDER BY RANDOM() LIMIT 5")
    examples = cur.fetchall()
    conn.close()

    for i, (name, dob, score) in enumerate(examples, 1):
        parts = name.split()
        lname = parts[0]
        fname = " ".join(parts[1:])
        y, m, d = dob.split("-")
        print(f"{i}. {lname} {fname}")
        print(f"   Дата: {d}.{m}.{y} (Скор: {int(score)})")
