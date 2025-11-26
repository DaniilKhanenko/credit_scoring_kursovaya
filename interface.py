from calculator import CalculatorTab
from history import HistoryTab
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оценка кредитоспособности")
        self.setGeometry(100, 100, 1100, 850)
        self.setStyleSheet("font-family: Segoe UI; font-size: 13px;")
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.calc_tab = CalculatorTab()
        self.hist_tab = HistoryTab()
        
        self.tabs.addTab(self.calc_tab, "Калькулятор")
        self.tabs.addTab(self.hist_tab, "История")

        self.tabs.currentChanged.connect(lambda i: self.hist_tab.load_data() if i==1 else None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
