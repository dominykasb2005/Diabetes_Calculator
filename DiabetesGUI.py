import tkinter as tk
from tkinter import messagebox
from DecTreeDiabetes import DecTreeDiabetes
import matplotlib.pyplot as plt

#colours for proj
c_dark = "#264653"
c_teal = "#2a9d8f"
c_gold = "#e9c46a"
c_bg = "#f2fbfa"
c_card_text = "#012a2a"
c_on_dark = "#ffffff"

#model instance
model = DecTreeDiabetes(csv_path="diabetes_data.csv", target_col="Diabetes")

root = tk.Tk()
root.title("Assignment 2 - A00321733")
root.geometry("1000x650")
root.configure(bg=c_bg)

header = tk.Label(root, text="Diabetes Prediction System", bg=c_dark,
fg=c_on_dark, font=("Helvetica", 20, "bold"), padx=12, pady=8)
header.place(relx=0.02, rely=0.01, relwidth=0.96)

left_frame = tk.Frame(root, bg=c_teal)
left_frame.place(relx=0.02, rely=0.12, relwidth=0.46, relheight=0.86)

right_frame = tk.Frame(root, bg=c_gold)
right_frame.place(relx=0.50, rely=0.12, relwidth=0.48, relheight=0.86)

left_title = tk.Label(left_frame, text="Model Controls", bg=c_teal, fg=c_on_dark,
font=("Helvetica", 14, "bold"))
left_title.pack(pady=(12, 6))

controls = tk.Frame(left_frame, bg=c_teal)
controls.pack(pady=6)

seed_var = tk.StringVar(value="1")
tk.Label(controls, text="seed", bg=c_teal, fg=c_on_dark).grid(row=0, column=0, padx=(4,4))
seed_menu = tk.OptionMenu(controls, seed_var, "1", "2", "3", "4", "5")
seed_menu.config(bg=c_dark, fg=c_on_dark, width=8)
seed_menu.grid(row=0, column=1, padx=(0,8))

train_btn = tk.Button(controls, text="TRAIN", bg=c_dark, fg=c_on_dark, width=10)
train_btn.grid(row=0, column=2, padx=6)

test_btn = tk.Button(controls, text="RUN MODEL", bg=c_on_dark, fg=c_teal, width=10)
test_btn.grid(row=0, column=3, padx=6)

fi_btn = tk.Button(controls, text="BAR CHART", bg=c_dark, fg=c_on_dark, width=16)
fi_btn.grid(row=0, column=4, padx=6)

res_box = tk.Frame(left_frame, bg=c_teal)
res_box.pack(padx=12, pady=(8,12), fill="both", expand=True)

res_text = tk.Text(res_box, height=18, bg=c_bg, fg=c_card_text, wrap="word")
res_text.pack(fill="both", expand=True, padx=6, pady=6)

detected = ", ".join(model.features_present)
missing = ", ".join(model.features_missing)
res_text.insert(tk.END, f"FEATURES DETECTED: {detected}\n")
if model.features_missing:
    res_text.insert(tk.END, f"Missing Features: {missing}\n")
res_text.insert(tk.END, "\n")

right_title = tk.Label(right_frame, text="Patient Details", bg=c_gold, fg=c_card_text,
font=("Helvetica", 14, "bold"))
right_title.pack(pady=(12, 6))

inputs_frame = tk.Frame(right_frame, bg=c_gold)
inputs_frame.pack(pady=6)

labels = [
    ("Age", "Age"),
    ("Gender", "Gender"),
    ("Polydipsia (excessive thirst)", "Polydipsia"),
    ("Sudden Weight Loss", "Sudden_Weight_Loss"),
    ("Fatigue", "Fatigue"),
    ("Polyphagia (hunger)", "Polyphagia"),
    ("Blurred Vision", "Blurred_Vision"),
    ("Muscle Stiffness", "Muscle_Stiffness"),
    ("Obesity", "Obesity")
]

widgets = {}
for i, (label_text, canonical) in enumerate(labels):
    row = tk.Frame(inputs_frame, bg=c_gold)
    row.grid(row=i, column=0, pady=4, padx=8, sticky="w")

    l = tk.Label(row, text=label_text, width=24, anchor="w", bg=c_gold, fg=c_card_text)
    l.pack(side="left")

    if canonical == "Age":
        e = tk.Entry(row, width=10)
        e.pack(side="right")
        widgets[canonical] = e
    elif canonical == "Gender":
        var = tk.StringVar(value="male")
        om = tk.OptionMenu(row, var, "female", "male")
        om.config(width=10)
        om.pack(side="right")
        widgets[canonical] = var
    else:
        var = tk.StringVar(value="no")
        om = tk.OptionMenu(row, var, "no", "yes")
        om.config(width=10)
        om.pack(side="right")
        widgets[canonical] = var

predict_btn = tk.Button(right_frame, text="Predict Result", bg=c_dark, fg=c_on_dark, width=18)
predict_btn.pack(pady=(14, 6))

result_frame = tk.Frame(right_frame, bg=c_gold)
result_frame.pack(pady=6)
tk.Label(result_frame, text="Result: ", bg=c_gold, fg=c_card_text).pack(side="left", padx=(2,8))
result_entry = tk.Entry(result_frame, width=24)
result_entry.pack(side="left")

def _map_widget_value(canonical):
    w = widgets.get(canonical)
    if canonical == "Age":
        val = w.get().strip()
        try:
            return int(float(val))
        except Exception:
            return 0
    else:
        v = w.get().strip().lower()
        if v in ("yes", "y", "true"):
            return 1
        if v in ("no", "n", "false"):
            return 0
        if v in ("male", "m"):
            return 1
        if v in ("female", "f"):
            return 0
        try:
            return int(float(v))
        except Exception:
            return 0

def on_train():
    try:
        model.update_seed(int(seed_var.get()))
        model.train(test_size=0.2)
        res_text.insert(tk.END, "Training Complete.\n")
        res_text.insert(tk.END, f"Features used for training: {', '.join(model.X.columns.tolist())}\n")
        unique_after = sorted(model.df[model.target].unique().tolist())
        if len(unique_after) < 2:
            res_text.insert(tk.END, "Warning: Single label is present in the dataset.\n")
            res_text.insert(tk.END, "Model will default to predicting the single class.\n\n")
    except Exception as e:
        res_text.insert(tk.END, f"Training Failed: {e}\n")

def on_test():
    try:
        cm, acc = model.test()
        res_text.insert(tk.END, "Test Results:\n")
        res_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")
        res_text.insert(tk.END, f"Accuracy: {int(acc*100)}%\n\n")
    except Exception as e:
        res_text.insert(tk.END, f"Testing Failed: {e}\n")

def on_predict():
    try:
        inputs_ordered = []
        for canon in model.canonical_features:
            if canon in widgets:
                inputs_ordered.append(_map_widget_value(canon))
            else:
                inputs_ordered.append(0)

        pred = model.predict(*inputs_ordered)
        result_entry.delete(0, tk.END)
        result_entry.insert(0, "Likely Diabetes" if pred == 1 else "Diabetes Not Likely")
    except Exception as e:
        result_entry.delete(0, tk.END)
        result_entry.insert(0, "error")
        res_text.insert(tk.END, f"prediction failed: {e}\n")

def on_feature_importance():
    try:
        imps = model.feature_importances()
        if not imps:
            res_text.insert(tk.END, "model not trained yet!!\n")
            return
        feats, vals = zip(*imps)
        fig, ax = plt.subplots(figsize=(8,4))
        colors = [c_dark if v == max(vals) else c_teal for v in vals]
        ax.barh(feats, vals, color=colors)
        ax.set_xlabel("Data Importance")
        ax.set_title("Data Importance")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        res_text.insert(tk.END, f"could not show data importance: {e}\n")

train_btn.config(command=on_train)
test_btn.config(command=on_test)
predict_btn.config(command=on_predict)
fi_btn.config(command=on_feature_importance)
root.mainloop()