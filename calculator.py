from model_inference import predict_approval_proba, get_neutral_score
import sqlite3
import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QGroupBox, QMessageBox, QComboBox, QScrollArea, QFrame, QSpinBox
)
from PyQt6.QtCore import Qt

DB_PATH = "data/credit_bureau.db"


GENDER_MAP = {"Мужской": "Male", "Женский": "Female"}
MARITAL_MAP = {"Холост/Не замужем": "Single", "Женат/Замужем": "Married", "Разведен(а)": "Divorced", "Вдовец/Вдова": "Widowed"}
EDU_MAP = {"Среднее": "High School", "Бакалавр": "Bachelor", "Магистр": "Master", "Кандидат наук": "PhD", "Другое": "Other"}
EMPLOY_MAP = {"Работает": "Employed", "Самозанятый": "Self-employed", "Безработный": "Unemployed", "Студент": "Student", "Пенсионер": "Retired"}
PURPOSE_MAP = {"Личные нужды": "Personal", "Недвижимость": "Home", "Автомобиль": "Car", "Образование": "Education", "Бизнес": "Business", "Другое": "Other"}

class CalculatorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_bureau_score = None
        self.last_calculation = None 

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.layout = QVBoxLayout(content)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # идентификация
        id_group = QGroupBox("1. Идентификация (БКИ)")
        id_l = QVBoxLayout()
        id_group.setLayout(id_l)

        name_row = QHBoxLayout()
        self.ln_input = QLineEdit()
        self.ln_input.setPlaceholderText("Иванов")
        self.fn_input = QLineEdit()
        self.fn_input.setPlaceholderText("Иван")
        name_row.addWidget(QLabel("Фамилия:"))
        name_row.addWidget(self.ln_input)
        name_row.addWidget(QLabel("Имя:"))
        name_row.addWidget(self.fn_input)
        id_l.addLayout(name_row)

        dr_row = QHBoxLayout()
        self.day = QSpinBox()
        self.day.setRange(1,31)
        self.day.setFixedWidth(50)
        self.month = QSpinBox()
        self.month.setRange(1,12)
        self.month.setFixedWidth(50)
        self.year = QSpinBox()
        self.year.setRange(1900,2025)
        self.year.setValue(1990)
        self.year.setFixedWidth(70)
        dr_row.addWidget(QLabel("ДР:"))
        dr_row.addWidget(self.day)
        dr_row.addWidget(QLabel("."))
        dr_row.addWidget(self.month)
        dr_row.addWidget(QLabel("."))
        dr_row.addWidget(self.year)
        dr_row.addStretch()
        id_l.addLayout(dr_row)

        btn_search = QPushButton("Найти историю")
        btn_search.clicked.connect(self.search_history)
        id_l.addWidget(btn_search)

        self.score_field = QLineEdit()
        self.score_field.setReadOnly(True)
        self.score_field.setStyleSheet("background: #eee; font-weight: bold;")
        r_sc = QHBoxLayout()
        r_sc.addWidget(QLabel("Рейтинг:"))
        r_sc.addWidget(self.score_field)
        id_l.addLayout(r_sc)
        self.layout.addWidget(id_group)

        # модель
        model_group = QGroupBox("2. Выбор модели")
        m_l = QHBoxLayout()
        model_group.setLayout(m_l)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["XGBoost", "Neural Network"])
        m_l.addWidget(QLabel("Алгоритм:"))
        m_l.addWidget(self.model_combo)
        self.layout.addWidget(model_group)

        # анкета
        p_group = QGroupBox("3. Личные данные")
        p_l = QVBoxLayout()
        p_group.setLayout(p_l)
        self.gender = self._combo(p_l, "Пол:", GENDER_MAP.keys())
        self.marital = self._combo(p_l, "Сем. положение:", MARITAL_MAP.keys())
        self.edu = self._combo(p_l, "Образование:", EDU_MAP.keys())
        self.job = self._combo(p_l, "Занятость:", EMPLOY_MAP.keys())
        self.layout.addWidget(p_group)

        # финансы
        f_group = QGroupBox("4. Финансы")
        f_l = QVBoxLayout()
        f_group.setLayout(f_l)
        self.income = self._inp(f_l, "Доход (год):", "1500000")
        self.loan = self._inp(f_l, "Сумма кредита:", "500000")
        self.purpose = self._combo(f_l, "Цель:", PURPOSE_MAP.keys())
        self.ex_loans = self._inp(f_l, "Активных кредитов:", "0")
        self.late = self._inp(f_l, "Просрочек (год):", "0")
        self.layout.addWidget(f_group)

        # кнопки
        self.btn_calc = QPushButton("РАССЧИТАТЬ ВЕРОЯТНОСТЬ")
        self.btn_calc.setStyleSheet("background-color: #27ae60; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        self.btn_calc.clicked.connect(self.calculate)
        self.layout.addWidget(self.btn_calc)

        btns_row = QHBoxLayout()
        self.btn_clear = QPushButton("Очистить поля")
        self.btn_clear.setStyleSheet("padding: 8px; background-color: #95a5a6; color: white; font-weight: bold;")
        self.btn_clear.clicked.connect(self.clear_fields)
        
        self.btn_save = QPushButton("Сохранить в историю")
        self.btn_save.setStyleSheet("padding: 8px; background-color: #2980b9; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_result_manually)
        self.btn_save.setEnabled(False)
        
        btns_row.addWidget(self.btn_clear)
        btns_row.addWidget(self.btn_save)
        self.layout.addLayout(btns_row)

        self.res_lbl = QLabel("Ожидание расчета...")
        self.res_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.res_lbl)

    def _inp(self, l, txt, ph):
        h = QHBoxLayout()
        lbl = QLabel(txt)
        lbl.setFixedWidth(150)
        inp = QLineEdit()
        inp.setPlaceholderText(ph)
        h.addWidget(lbl)
        h.addWidget(inp)
        l.addLayout(h)
        return inp

    def _combo(self, l, txt, items):
        h = QHBoxLayout()
        lbl = QLabel(txt)
        lbl.setFixedWidth(150)
        cmb = QComboBox()
        cmb.addItems(list(items))
        h.addWidget(lbl)
        h.addWidget(cmb)
        l.addLayout(h)
        return cmb

    def get_dob_str(self):
        try:
            d = datetime.date(self.year.value(), self.month.value(), self.day.value())
            if d > datetime.date.today() or d.year < 1900: raise ValueError
            return d.strftime("%Y-%m-%d")
        except: return None

    def get_fio(self):
        ln, fn = self.ln_input.text().strip(), self.fn_input.text().strip()
        return f"{ln} {fn}" if ln and fn else None

    def search_history(self):
        fio, dob = self.get_fio(), self.get_dob_str()
        if not fio or not dob: 
            QMessageBox.warning(self, "Ошибка", "Введите ФИО и дату!")
            return
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT credit_score FROM credit_scores WHERE full_name=? AND birth_date=?", (fio, dob))
            row = cur.fetchone()
            conn.close()
            if row:
                self.current_bureau_score = row[0]
                self.score_field.setText(str(int(row[0])))
                QMessageBox.information(self, "Успех", f"Скор: {int(row[0])}")
            else:
                self.current_bureau_score = None
                self.score_field.setText("НЕТ ДАННЫХ")
                QMessageBox.warning(self, "Инфо", "Клиент не найден.")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    def clear_fields(self):
        self.ln_input.clear()
        self.fn_input.clear()
        self.year.setValue(1990)
        self.month.setValue(1)
        self.day.setValue(1)
        self.current_bureau_score = None
        self.score_field.clear()
        self.gender.setCurrentIndex(0)
        self.marital.setCurrentIndex(0)
        self.edu.setCurrentIndex(0)
        self.job.setCurrentIndex(0)
        self.purpose.setCurrentIndex(0)
        self.income.clear()
        self.loan.clear()
        self.ex_loans.clear()
        self.late.clear()
        self.res_lbl.setText("Ожидание расчета...")
        self.res_lbl.setStyleSheet("")
        self.last_calculation = None
        self.btn_save.setEnabled(False)

    def calculate(self):
        try:
            dob = self.get_dob_str()
            if not dob: raise ValueError("Неверная дата")
            age = datetime.date.today().year - self.year.value()
            data = {
                "Age": age,
                "Gender": GENDER_MAP[self.gender.currentText()],
                "MaritalStatus": MARITAL_MAP[self.marital.currentText()],
                "EducationLevel": EDU_MAP[self.edu.currentText()],
                "EmploymentStatus": EMPLOY_MAP[self.job.currentText()],
                "AnnualIncome": float(self.income.text()),
                "LoanAmountRequested": float(self.loan.text()),
                "PurposeOfLoan": PURPOSE_MAP[self.purpose.currentText()],
                "CreditScore": self.current_bureau_score if self.current_bureau_score else get_neutral_score(),
                "ExistingLoansCount": int(self.ex_loans.text()),
                "LatePaymentsLastYear": int(self.late.text())
            }
            model_key = "xgb" if self.model_combo.currentIndex() == 0 else "nn"
            prob = predict_approval_proba(data, model_key) * 100
            
            color = "#27ae60" if prob > 70 else "#f39c12" if prob > 40 else "#c0392b"
            self.res_lbl.setText(f"Вероятность: {prob:.1f}%")
            self.res_lbl.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold; border: 2px solid {color}; padding: 10px; background: white;")

            self.last_calculation = {
                "fio": self.get_fio() or "Unknown", "dob": dob, "data": data,
                "model": model_key, "prob": prob, "score": data["CreditScore"]
            }
            self.btn_save.setEnabled(True)
            
        except ValueError: QMessageBox.warning(self, "Ошибка", "Проверьте числа и дату!")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    def save_result_manually(self):
        if not self.last_calculation: return
        try:
            c = self.last_calculation
            d = c["data"]
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO application_history (full_name, birth_date, age, gender, marital_status, education, employment, 
                 income, loan_amount, purpose, existing_loans, late_payments, model_used, predicted_probability, bureau_score_used)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", 
                (c["fio"], c["dob"], d["Age"], d["Gender"], d["MaritalStatus"], d["EducationLevel"], d["EmploymentStatus"],
                 d["AnnualIncome"], d["LoanAmountRequested"], d["PurposeOfLoan"], d["ExistingLoansCount"], d["LatePaymentsLastYear"],
                 c["model"], c["prob"]/100, c["score"]))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Успех", "Запись сохранена в историю!")
            self.btn_save.setEnabled(False)
        except Exception as e: QMessageBox.critical(self, "Ошибка сохранения", str(e))
