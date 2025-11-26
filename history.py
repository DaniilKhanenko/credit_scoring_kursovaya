import sqlite3
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, 
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

DB_PATH = "data/credit_bureau.db"

class HistoryTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self.load_data)
        self.btn_export = QPushButton("Экспорт (Всё)")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        btn_layout.addWidget(self.btn_refresh); btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["Дата", "ФИО", "Возраст", "Образ.", "Скор", "Доход", "Сумма", "Рез.", "Модель"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.load_data()

    def load_data(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("""
                SELECT request_date, full_name, age, education, bureau_score_used, income, loan_amount, predicted_probability, model_used 
                FROM application_history ORDER BY id DESC LIMIT 100
            """, conn)
            conn.close()

            self.table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.table.setItem(i, 0, QTableWidgetItem(str(row["request_date"])))
                self.table.setItem(i, 1, QTableWidgetItem(str(row["full_name"])))
                self.table.setItem(i, 2, QTableWidgetItem(str(row["age"])))
                self.table.setItem(i, 3, QTableWidgetItem(str(row["education"])))
                self.table.setItem(i, 4, QTableWidgetItem(str(int(row["bureau_score_used"]))))
                self.table.setItem(i, 5, QTableWidgetItem(f"{row['income']:,.0f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{row['loan_amount']:,.0f}"))
                
                prob = float(row["predicted_probability"]) * 100
                item_res = QTableWidgetItem(f"{prob:.1f}%")
                if prob > 70: item_res.setBackground(Qt.GlobalColor.green)
                elif prob < 40: item_res.setBackground(Qt.GlobalColor.red)
                self.table.setItem(i, 7, item_res)
                self.table.setItem(i, 8, QTableWidgetItem(str(row["model_used"])))
        except: pass

    def export_data(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM application_history", conn)
            conn.close()
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", "full_report.csv", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Успех", "Полный отчет сохранен!")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))
