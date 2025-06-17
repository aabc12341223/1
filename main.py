# =========================
# 1. THƯ VIỆN & DỮ LIỆU
# =========================
import sys
import pandas as pd
import joblib
from PyQt5.QtGui import QKeyEvent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt

df = pd.read_csv('Student Depression Dataset.csv',sep=';')

features = ["Age", "Gender", "CGPA", "Academic Pressure", "Study Satisfaction","Sleep Duration", "Financial Stress", "Family History of Mental Illness"]
target = "Depression"

# Tạo bản sao dữ liệu đầu vào và mục tiêu
X = df[features].copy()
y = df[target]
# Nhóm định lượng
numerical_cols = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Financial Stress"]
# Xử lý dữ liệu phân loại
X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
X["Family History of Mental Illness"] = X["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# Mã hoá sleep duration theo thứ tự
sleep_duration_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3
}
X["Sleep Duration"] = X["Sleep Duration"].map(sleep_duration_map)

# Kiểm tra dữ liệu bị thiếu
missing = X.isnull().sum()

# Điền giá trị thiếu bằng trung bình (cho numeric) nếu có
X.fillna(X.mean(numeric_only=True), inplace=True)

X.head(), missing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tránh cảnh báo khi gán
X_train = X_train.copy()
X_test = X_test.copy()

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Chuyển về float để dùng với SMOTE
X_train = X_train.astype(float)

# =========================
# 4. ÁP DỤNG SMOTE
# =========================

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# =========================
# 5. HUẤN LUYỆN & LƯU MÔ HÌNH
# =========================

model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Mô hình đã được huấn luyện và lưu thành công!")

# =========================
# 6. DỰ BÁO TỪ DỮ LIỆU NGƯỜI DÙNG
# =========================

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Load file .ui
        uic.loadUi("app.ui", self)
        #Cài đặt event cho edittext

        for widget in [self.textEdit, self.textEdit_3, self.textEdit_4, self.textEdit_5, self.textEdit_7]:
            widget.installEventFilter(self)
        #kết nối button với hàm xử lý sk
        self.pushButton.clicked.connect(self.on_button_click)

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Tab:
            self.focusNextChild()  # Di chuyển focus sang widget tiếp theo
            return True  # Chặn Tab mặc định
        return super().eventFilter(obj, event)

    def on_button_click(self):
        # hàm xử lý sự kiện khi bấm nút
        edits = [
            self.textEdit, self.textEdit_3, self.textEdit_4, self.textEdit_5, self.textEdit_7
        ]
        if any(edit.toPlainText().strip() == "" for edit in edits):
            QtWidgets.QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ tất cả các ô.")
            return
        for edit in edits:
            text = edit.toPlainText().strip()
            if not text.isdigit():
                QtWidgets.QMessageBox.warning(self, "Lỗi dữ liệu", "Chỉ được nhập số trong các ô.")
                return

        #Lấy dữ liệu từ editText và comboBox
        age = int(self.textEdit.toPlainText())
        gender = self.comboBox_2.currentText()
        cgpa = float(self.textEdit_3.toPlainText())
        academic_pressure = int(self.textEdit_4.toPlainText())
        study_satisfaction = int(self.textEdit_5.toPlainText())
        sleep_duration = self.comboBox.currentText()
        financial_stress = int(self.textEdit_7.toPlainText())
        family_history = self.comboBox_3.currentText()

        #Xử lý đầu vào người dùng
        sleep_map = {
            "Less than 5 hours": 0,
            "5-6 hours": 1,
            "7-8 hours": 2,
            "More than 8 hours": 3
        }
        gender_map = {"Male": 1, "Female": 0}
        history_map = {"Yes": 1, "No": 0}

        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender_map.get(gender, 0),
            "CGPA": cgpa,
            "Academic Pressure": academic_pressure,
            "Study Satisfaction": study_satisfaction,
            "Sleep Duration": sleep_map.get(sleep_duration, 1),
            "Financial Stress": financial_stress,
            "Family History of Mental Illness": history_map.get(family_history, 0)
        }])

        # Chuẩn hóa dữ liệu đầu vào
        scaler = joblib.load("scaler.pkl")
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Dự đoán với mô hình đã huấn luyện
        model = joblib.load("logistic_model.pkl")
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        noi_dung = f"=== KẾT QUẢ DỰ BÁO ===\nTình trạng: {'Có nguy cơ mắc bệnh trầm cảm' if prediction == 1 else 'Bình thường'}\nXác suất: {probability:.2%}"
        self.label_8.setText(noi_dung)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
