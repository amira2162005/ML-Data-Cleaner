import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = None
model = None
X_train, X_test, y_train, y_test = None, None, None, None
input_entries = {}

root = tk.Tk()
root.title("ML Analyzer")
root.geometry("1000x700")
root.configure(bg="#FFF0F5")

# ---------- PAGE 1 -----00000000000000000000000000---
page1 = tk.Frame(root, bg="#FFF0F5")
page1.place(relwidth=1, relheight=1)

title_label = tk.Label(page1, text="ML Analyzer: Predict & Classify Any Dataset",
                       font=("Helvetica", 24, "bold"), fg="#DB7093", bg="#FFF0F5")
title_label.pack(expand=True)

def go_to_page2():
    page1.place_forget()
    page2.place(relwidth=1, relheight=1)

start_button = tk.Button(page1, text="Enter with your right foot", command=go_to_page2,
                         bg="#DB7093", fg="white", font=("Helvetica", 14), width=25, height=2)
start_button.pack(pady=20)

# ---------- PAGE 2 ----------
page2 = tk.Frame(root, bg="#FFF0F5")

def go_to_page1():
    page2.place_forget()
    page1.place(relwidth=1, relheight=1)

def go_to_page3():
    page2.place_forget()
    page3.place(relwidth=1, relheight=1)
    update_target_dropdown()

back_button_p2 = tk.Button(page2, text="← Back", command=go_to_page1,
                           bg="#DB7093", fg="white", font=("Helvetica", 10), width=8)
back_button_p2.place(x=10, y=10)

label_page2 = tk.Label(page2, text="Processing Data", font=("Helvetica", 24, "bold"),
                       fg="#DB7093", bg="#FFF0F5")
label_page2.pack(pady=30)

btn_frame = tk.Frame(page2, bg="#FFF0F5")
btn_frame.pack(pady=10)

button_style = {"bg": "#DB7093", "fg": "white", "font": ("Helvetica", 10), "width": 20}

def upload_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        messagebox.showinfo("Success", "File loaded successfully")
        clear_table()
        clear_process()

upload_btn = tk.Button(btn_frame, text="Upload CSV", command=upload_csv, **button_style)
upload_btn.pack(side=tk.LEFT, padx=5)

def show_dataframe(dataframe):
    clear_process()
    for widget in table_frame.winfo_children():
        widget.destroy()
    if dataframe is not None:
        tree = ttk.Treeview(table_frame, show='headings')
        tree["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        for index, row in dataframe.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill="both", expand=True)

def show_head():
    if df is not None:
        show_dataframe(df.head())
        messagebox.showinfo("Info", "Displayed first 5 rows")
    else:
        messagebox.showwarning("Warning", "Please upload CSV first!")

def show_columns():
    clear_process()
    for widget in table_frame.winfo_children():
        widget.destroy()
    if df is not None:
        cols = list(df.columns)
        label = tk.Label(table_frame, text="\n".join(cols), font=("Courier", 10), bg="white", anchor="w", justify="left")
        label.pack(fill="both", expand=True)
        messagebox.showinfo("Info", "Displayed column names")
    else:
        messagebox.showwarning("Warning", "Please upload CSV first!")

def describe_data():
    if df is not None:
        show_dataframe(df.describe())
        messagebox.showinfo("Info", "Displayed data description")
    else:
        messagebox.showwarning("Warning", "Please upload CSV first!")

def clear_table():
    for widget in table_frame.winfo_children():
        widget.destroy()

def clear_process():
    for widget in process_frame.winfo_children():
        widget.destroy()
def handle_missing_values():
    global df
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
    return True


def remove_outliers():
    global df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return True

def encode_categorical():
    global df
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return True

def normalize_features():
    global df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return True

def processing_action():
    if df is None:
        messagebox.showwarning("Warning", "Please upload CSV file first!")
        return

    clear_process()
    ops = [
        ("Handle Missing Values", handle_missing_values),
        ("Remove Outliers", remove_outliers),
        ("Encoding Categorical Data", encode_categorical),
        ("Normalize Features", normalize_features)
    ]

    for op_name, op_func in ops:
        frame = tk.Frame(process_frame, bg="#FFF0F5")
        frame.pack(fill="x", padx=10, pady=5)
        label = tk.Label(frame, text=op_name, font=("Arial", 12), bg="#FFF0F5")
        label.pack(side="left")
        success = op_func()
        if success:
            check_label = tk.Label(frame, text="✔", fg="green", font=("Arial", 14), bg="#FFF0F5")
            check_label.pack(side="right")

    show_dataframe(df.head())
    messagebox.showinfo("Processing", "All processing steps completed!")

show_head_btn = tk.Button(btn_frame, text="Show First 5 Rows", command=show_head, **button_style)
show_head_btn.pack(side=tk.LEFT, padx=5)

show_cols_btn = tk.Button(btn_frame, text="Show Column Names", command=show_columns, **button_style)
show_cols_btn.pack(side=tk.LEFT, padx=5)

describe_btn = tk.Button(btn_frame, text="Describe Data", command=describe_data, **button_style)
describe_btn.pack(side=tk.LEFT, padx=5)

processing_btn = tk.Button(btn_frame, text="Processing", command=processing_action, **button_style)
processing_btn.pack(side=tk.LEFT, padx=5)

table_frame = tk.Frame(page2, bg="white")
table_frame.pack(fill="both", expand=True, padx=10, pady=10)

process_frame = tk.Frame(page2, bg="#FFF0F5")
process_frame.pack(fill="x", padx=10, pady=10)

continue_button = tk.Button(page2, text="Continue", command=go_to_page3,
                             bg="#DB7093", fg="white", font=("Helvetica", 14), width=20, height=2)
continue_button.pack(side="bottom", pady=20)

# ---------- PAGE 3 ----------import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# إنشاء صفحة 3
page3 = tk.Frame(root, bg="#FFF0F5")

def go_to_page2_from_3():
    page3.place_forget()
    page2.place(relwidth=1, relheight=1)

back_button_p3 = tk.Button(page3, text="← Back", command=go_to_page2_from_3,
                           bg="#DB7093", fg="white", font=("Helvetica", 10), width=8)
back_button_p3.place(x=10, y=10)

label_page3 = tk.Label(page3, text="Model Training & Prediction", font=("Helvetica", 24, "bold"),
                       fg="#DB7093", bg="#FFF0F5")
label_page3.pack(pady=20)

# ---------- Task Type ----------
task_var = tk.StringVar()
tk.Label(page3, text="Select Task Type:", font=("Helvetica", 12), bg="#FFF0F5").pack()
task_dropdown = ttk.Combobox(page3, textvariable=task_var, state="readonly")
task_dropdown['values'] = ["Classification", "Regression"]
task_dropdown.pack(pady=5)

# ---------- Target Column ----------
target_var = tk.StringVar()
tk.Label(page3, text="Select Target Column:", font=("Helvetica", 12), bg="#FFF0F5").pack()
target_dropdown = ttk.Combobox(page3, textvariable=target_var, state="readonly")
target_dropdown.pack(pady=5)

# ---------- Model ----------
model_var = tk.StringVar()
tk.Label(page3, text="Select Model:", font=("Helvetica", 12), bg="#FFF0F5").pack()
model_dropdown = ttk.Combobox(page3, textvariable=model_var, state="readonly")
model_dropdown.pack(pady=5)

# تحديث القوائم حسب نوع المهمة
def update_target_dropdown():
    if df is not None:
        target_dropdown['values'] = list(df.columns)

def update_models(event=None):
    if task_var.get() == "Classification":
        model_dropdown['values'] = ['KNN', 'SVM', 'Decision Tree']
    elif task_var.get() == "Regression":
        model_dropdown['values'] = ['Linear Regression']

task_dropdown.bind("<<ComboboxSelected>>", update_models)

# ---------- وظائف التقسيم والتدريب والتنبؤ ----------
def split_data():
    global X_train, X_test, y_train, y_test
    if df is None or target_var.get() == "":
        messagebox.showwarning("Warning", "Please select a target column first.")
        return

    try:
        split_ratio = simpledialog.askfloat("Split Ratio", "Enter test size (e.g. 0.2):", minvalue=0.05, maxvalue=0.95)
        if split_ratio is None:
            return

        target = target_var.get()
        X = df.drop(columns=[target])
        y = df[target]

        if y.isnull().any() or X.isnull().any().any():
            messagebox.showerror("Error", "Data contains missing values. Please preprocess first.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
        messagebox.showinfo("Success", f"data has been spilited successfully .")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def train_model():
    global model
    try:
        if X_train is None or y_train is None:
            messagebox.showwarning("Warning", "Please split the data first.")
            return

        selected_model = model_var.get()
        if selected_model == "KNN":
            model = KNeighborsClassifier()
        elif selected_model == "SVM":
            model = SVC()
        elif selected_model == "Decision Tree":
            model = DecisionTreeClassifier()
        elif selected_model == "Linear Regression":
            model = LinearRegression()
        else:
            messagebox.showerror("Error", "Please select a model.")
            return

        model.fit(X_train, y_train)
        messagebox.showinfo("Success", "model has been trainerd successfully.\nالنتائج ستُعرض في الصفحة الرابعة)")
    except Exception as e:
        messagebox.showerror("Error", str(e))
       
class PredictPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        ttk.Label(self, text="Manual Input for Prediction", font=("Arial", 18)).pack(pady=20)
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(pady=10)
        self.predict_button = ttk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        self.output_label = ttk.Label(self, text="")
        self.output_label.pack(pady=10)
        self.bind("<<ShowFrame>>", self.on_show_frame)

    def on_show_frame(self, event=None):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.controller.input_entries = []
        if self.controller.df is not None:
            columns = self.controller.df.columns[:-1]
            for col in columns:
                ttk.Label(self.input_frame, text=col).pack()
                entry = ttk.Entry(self.input_frame)
                entry.pack()
                self.controller.input_entries.append(entry)
                
def predict_model():
    try:
        if model is None or X_test is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        predictions = model.predict(X_test)
        # هنا تُخزن النتائج للعرض في الصفحة الرابعة
        global last_predictions
        last_predictions = predictions
        messagebox.showinfo("Success", "تم إجراء التنبؤ.\n(النتائج ستُعرض في الصفحة الرابعة)")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        

# ---------- الأزرار: بجانب بعض ----------
buttons_frame = tk.Frame(page3, bg="#FFF0F5")
buttons_frame.pack(pady=15)

split_btn = tk.Button(buttons_frame, text="Split", command=split_data,
                      bg="#DB7093", fg="white", font=("Helvetica", 12), width=10)
split_btn.grid(row=0, column=0, padx=5)

train_btn = tk.Button(buttons_frame, text="Train", command=train_model,
                      bg="#DB7093", fg="white", font=("Helvetica", 12), width=10)
train_btn.grid(row=0, column=1, padx=5)

predict_btn = tk.Button(buttons_frame, text="Predict", command=predict_model,
                        bg="#DB7093", fg="white", font=("Helvetica", 12), width=10)
predict_btn.grid(row=0, column=2, padx=5)

# زر الذهاب للصفحة الرابعة لعرض النتائج (أنت تنشئه حسب تصميمك)
#go_to_page4_btn = tk.Button(page3, text="↦ Go to Results", command=lambda: show_results_in_page4(),
                   #         bg="#DB7093", fg="white", font=("Helvetica", 12), width=20)
#go_to_page4_btn.pack(pady=10)
# ---------- Manual Prediction ----------
manual_entries = {}

def generate_manual_inputs():
    global manual_entries
    manual_entries = {}

    if df is None:
        messagebox.showwarning("Warning", "Please upload data first.")
        return

    if target_var.get() == "":
        messagebox.showwarning("Warning", "Please select target column first.")
        return

    input_window = tk.Toplevel(root)
    input_window.title("Manual Input")
    input_window.geometry("400x600")
    
    tk.Label(input_window, text="Enter feature values:", font=("Helvetica", 14)).pack(pady=10)

    features = df.drop(columns=[target_var.get()]).columns
    for feat in features:
        frame = tk.Frame(input_window)
        frame.pack(pady=2)
        label = tk.Label(frame, text=feat, width=20, anchor="w")
        label.pack(side=tk.LEFT)
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        manual_entries[feat] = entry

    def predict_manual():
        try:
            input_data = [float(manual_entries[feat].get()) for feat in features]
            prediction = model.predict([input_data])[0]
            messagebox.showinfo("Prediction Result", f"Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    predict_btn = tk.Button(input_window, text="Predict", command=predict_manual,
                            bg="#DB7093", fg="white", font=("Helvetica", 12))
    predict_btn.pack(pady=10)

manual_input_btn = tk.Button(page3, text="Manual Predict", command=generate_manual_inputs,
                             bg="#DB7093", fg="white", font=("Helvetica", 12), width=20)
manual_input_btn.pack(pady=10)


# ---------- Go to Page 4 ----------
# ---------- PAGE 4 ----------
page4 = tk.Frame(root, bg="#FFF0F5")

def go_to_page3_from_4():
    page4.place_forget()
    page3.place(relwidth=1, relheight=1)

back_button_p4 = tk.Button(page4, text="← Back", command=go_to_page3_from_4,
                           bg="#DB7093", fg="white", font=("Helvetica", 10), width=8)
back_button_p4.place(x=10, y=10)

label_page4 = tk.Label(page4, text="Model Evaluation", font=("Helvetica", 24, "bold"),
                       fg="#DB7093", bg="#FFF0F5")
label_page4.pack(pady=20)

eval_text = tk.Text(page4, width=80, height=15, font=("Helvetica", 12))
eval_text.pack(pady=10, padx=10)

def show_evaluation():
    if model is None or X_test is None or y_test is None:
        messagebox.showwarning("Warning", "Please train the model first.")
        return

    eval_text.delete('1.0', tk.END)

    predictions = model.predict(X_test)

    if task_var.get() == "Classification":
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        eval_text.insert(tk.END, f"Accuracy: {acc:.2f}\n")
        eval_text.insert(tk.END, f"Precision: {prec:.2f}\n")
        eval_text.insert(tk.END, f"Recall: {recall:.2f}\n")
        eval_text.insert(tk.END, f"F1 Score: {f1:.2f}\n\n")
        eval_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")

        # Show confusion matrix plot
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    elif task_var.get() == "Regression":
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        eval_text.insert(tk.END, f"Mean Absolute Error (MAE): {mae:.2f}\n")
        eval_text.insert(tk.END, f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        eval_text.insert(tk.END, f"R² Score: {r2:.2f}\n")

        # Show scatter plot of predictions vs actual
        plt.scatter(y_test, predictions, alpha=0.7)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()

eval_btn = tk.Button(page4, text="Show Evaluation", command=show_evaluation,
                     bg="#A2526E", fg="white", font=("Helvetica", 12), width=20)
eval_btn.pack(pady=10)

# -------- Add button to go page4 from page3 ----------
def go_to_page4():
    page3.place_forget()
    page4.place(relwidth=1, relheight=1)

go_eval_button = tk.Button(page3, text="Go to Evaluation Page", command=go_to_page4,
                           bg="#DB7093", fg="white", font=("Helvetica", 12), width=20)
go_eval_button.pack(pady=10)
from sklearn.cluster import KMeans

# --- إنشاء صفحة 5 ---

page5 = tk.Frame(root, bg="#FFF0F5")

def go_to_page3_from_5():
    page5.place_forget()
    page4.place(relwidth=1, relheight=1)

back_button_p5 = tk.Button(page5, text="← Back", command=go_to_page3_from_5,
                          bg="#DB7093", fg="white", font=("Helvetica", 10), width=8)
back_button_p5.pack(anchor="nw", padx=10, pady=10)

label_page5 = tk.Label(page5, text="Clustering (KMeans)", font=("Helvetica", 24, "bold"),
                      fg="#DB7093", bg="#FFF0F5")
label_page5.pack(pady=20)

# إدخال عدد الكلستر
clusters_label = tk.Label(page5, text="Enter number of clusters:", font=("Helvetica", 12), bg="#FFF0F5")
clusters_label.pack()

clusters_entry = tk.Entry(page5, font=("Helvetica", 12))
clusters_entry.pack(pady=5)

# إطار لعرض النتائج
result_frame = tk.Frame(page5, bg="white", height=400)
result_frame.pack(fill="both", expand=True, padx=10, pady=10)

def run_kmeans():
    for widget in result_frame.winfo_children():
        widget.destroy()

    if df is None:
        messagebox.showwarning("Warning", "Please upload and preprocess data first!")
        return

    try:
        k = int(clusters_entry.get())
        if k <= 0:
            messagebox.showerror("Error", "Number of clusters must be positive!")
            return
    except:
        messagebox.showerror("Error", "Please enter a valid integer for clusters!")
        return

    # نستخدم كل الأعمدة الرقمية فقط للتجميع
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        messagebox.showwarning("Warning", "Need at least 2 numeric features for clustering visualization.")
        return

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)

    # نضيف عمود الكلستر لبياناتنا مؤقتاً للعرض
    numeric_df["Cluster"] = clusters

    # عرض النتائج في جدول
    tree = ttk.Treeview(result_frame, columns=list(numeric_df.columns), show='headings')
    for col in numeric_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    for idx, row in numeric_df.iterrows():
        tree.insert("", "end", values=list(row))
    tree.pack(fill="both", expand=True)

    # رسم Scatter plot لأول عمودين مع تلوين حسب الكلستر
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=clusters, palette="Set1")
    plt.title("KMeans Clustering")
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])

    plt.tight_layout()
    plt.show()

# زر تشغيل التجميع
run_btn = tk.Button(page5, text="Run KMeans", command=run_kmeans,
                   bg="#DB7093", fg="white", font=("Helvetica", 14), width=15, height=2)
run_btn.pack(pady=15)


def go_to_page5():
    page4.place_forget()
    page5.place(relwidth=1, relheight=1)

go_eval_button = tk.Button(page4, text="Go to cluster", command=go_to_page5,
                           bg="#DB7093", fg="white", font=("Helvetica", 12), width=20)
go_eval_button.pack(pady=10)
# ---------- START ----------
page1.place(relwidth=1, relheight=1)
root.mainloop()
