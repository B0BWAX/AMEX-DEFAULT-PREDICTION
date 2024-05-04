import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pickle
import xgboost as xgb

class CreditPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Prediction")

        # Frame for file selection
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack(pady=10)

        self.file_label = tk.Label(self.file_frame, text="Select CSV File:")
        self.file_label.grid(row=0, column=0)

        self.file_path = tk.StringVar()
        self.file_entry = tk.Entry(self.file_frame, textvariable=self.file_path, width=50)
        self.file_entry.grid(row=0, column=1)

        self.browse_button = tk.Button(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2)

        # Frame for model selection
        self.model_frame = tk.Frame(self.root)
        self.model_frame.pack(pady=10)

        self.model_label = tk.Label(self.model_frame, text="Select Prediction Model:")
        self.model_label.grid(row=0, column=0)

        self.selected_model = tk.StringVar()
        self.model_optionmenu = tk.OptionMenu(self.model_frame, self.selected_model, "ANN", "XGB")
        self.model_optionmenu.grid(row=0, column=1)

        # Calculate button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)

    def feature_engineer(self, df):
        features = [c for c in list(df.columns) if c not in ['customer_ID','S_2']] 
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in features if col not in cat_features]

        test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

        test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

        df = pd.concat([test_num_agg, test_cat_agg], axis=1)
        del test_num_agg, test_cat_agg
        
        return df

    def predict(self):
        file_path = self.file_path.get()
        selected_model = self.selected_model.get()

        try:
            df = pd.read_csv(file_path)
            engineered_df = self.feature_engineer(df)
            engineered_df = engineered_df.fillna(-127)
            if selected_model == "ANN":
                model_path = 'models/AMEX_ANN_Model.pkl'
                model = pickle.load(open(model_path, 'rb'))
                prediction = model.predict(engineered_df.head(1))
            if selected_model == "XGB":
                model_path = 'models/AMEX_XGB_Model.pkl'
                model = pickle.load(open(model_path, 'rb'))
                d_engineered_df = xgb.DMatrix(engineered_df)
                prediction = model.predict(d_engineered_df)
            if prediction < 0.5:
                messagebox.showinfo("Prediction", f"Prediction: {prediction} \n Low likelihood of default")
            else:
                messagebox.showinfo("Prediction", f"Prediction: {prediction} \n High likelihood of default")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CreditPredictionApp(root)
    root.mainloop()
