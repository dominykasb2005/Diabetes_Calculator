import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class DecTreeDiabetes:
    def __init__(self, csv_path="diabetes_data.csv", target_col="Diabetes"):
        self.raw = pd.read_csv(csv_path)

        self.canonical_features = [
            "Age", "Gender", "Polydipsia", "Sudden_Weight_Loss",
            "Fatigue", "Polyphagia", "Blurred_Vision",
            "Muscle_Stiffness", "Obesity"
        ]
        self.target = target_col

        if self.target not in self.raw.columns:
            raise ValueError(f"target column '{self.target}' not found in csv")

        self.aliases = {
            "Age": ["Age", "age"],
            "Gender": ["Gender", "gender"],
            "Polydipsia": ["Polydipsia", "polydipsia", "ExcessThirst"],
            "Sudden_Weight_Loss": ["Sudden_Weight_Loss", "WeightLossSudden", "WeightLoss_Sudden", "WeightLossSudden"],
            "Fatigue": ["Fatigue", "fatigue"],
            "Polyphagia": ["Polyphagia", "polyphagia"],
            "Blurred_Vision": ["Blurred_Vision", "BlurredVision", "Blurredvision"],
            "Muscle_Stiffness": ["Muscle_Stiffness", "MuscleStiffness", "MuscleStiff"],
            "Obesity": ["Obesity", "obesity"]
        }

        self.col_map = {}
        self.features_present = []
        self.features_missing = []
        for canon in self.canonical_features:
            found = None
            for alias in self.aliases.get(canon, []):
                if alias in self.raw.columns:
                    found = alias
                    break
            if found:
                self.col_map[canon] = found
                self.features_present.append(canon)
            else:
                self.features_missing.append(canon)

        actual_cols = [self.col_map[c] for c in self.features_present] + [self.target]
        self.df = self.raw[actual_cols].copy()

        rename_map = {self.col_map[c]: c for c in self.features_present}
        self.df.rename(columns=rename_map, inplace=True)

        self._map_target_values()
        self._preprocess_dataset()

        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

        self.seed = 1
        self.tree = DecisionTreeClassifier()
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def _map_target_values(self):
        raw_target = self.df[self.target].astype(str).str.strip()
        lowered = raw_target.str.lower()

        map_1_keywords = {"1", "yes", "y", "true", "positive", "pos", "diabetic", "tested_positive"}
        map_0_keywords = {"0", "no", "n", "false", "negative", "neg", "non-diabetic", "notdiabetic", "tested_negative"}

        mapped = []
        for v, low in zip(raw_target, lowered):
            if low in map_1_keywords:
                mapped.append(1)
            elif low in map_0_keywords:
                mapped.append(0)
            else:
                try:
                    num = int(float(v))
                    mapped.append(1 if num == 1 else 0)
                except Exception:
                    mapped.append(0)

        self.df[self.target] = pd.Series(mapped, index=self.df.index).astype(int)
        self.unique_labels = sorted(self.df[self.target].unique().tolist())

    def _to_binary_series(self, ser):
        if ser.dtype == object:
            s = ser.str.strip().str.lower()
            mapping = {
                "yes": 1, "y": 1, "true": 1, "1": 1,
                "no": 0, "n": 0, "false": 0, "0": 0,
                "male": 1, "m": 1, "man": 1,
                "female": 0, "f": 0, "woman": 0
            }
            mapped = s.map(mapping)
            if mapped.notna().sum() >= int(0.5 * len(mapped)):
                return mapped.fillna(0).astype(int)
            else:
                return pd.to_numeric(ser, errors="coerce").fillna(0).astype(int)
        else:
            return pd.to_numeric(ser, errors="coerce").fillna(0).astype(int)

    def _preprocess_dataset(self):
        for col in self.df.columns:
            if col == "Age":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)
            elif col == self.target:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)
            else:
                self.df[col] = self._to_binary_series(self.df[col])

    def update_seed(self, new_seed):
        self.seed = int(new_seed)

    def train(self, test_size=0.2):
        strat = self.y if len(set(self.y)) > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.seed, stratify=strat
        )
        self.tree = DecisionTreeClassifier()
        self.tree.fit(self.X_train, self.y_train)

    def test(self):
        y_hat = self.tree.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_hat, labels=[0, 1])
        acc = accuracy_score(self.y_test, y_hat)
        return cm, acc

    def read_categories(self):
        return "0 = no diabetes / 1 = diabetes"

    def feature_importances(self):
        if not hasattr(self.tree, "feature_importances_"):
            return []
        imps = list(self.tree.feature_importances_)
        return list(zip(self.X.columns.tolist(), imps))

    def predict(self, Age, Gender, Polydipsia, Sudden_Weight_Loss, Fatigue,
                Polyphagia, Blurred_Vision, Muscle_Stiffness, Obesity):
        inputs = {
            "Age": Age,
            "Gender": Gender,
            "Polydipsia": Polydipsia,
            "Sudden_Weight_Loss": Sudden_Weight_Loss,
            "Fatigue": Fatigue,
            "Polyphagia": Polyphagia,
            "Blurred_Vision": Blurred_Vision,
            "Muscle_Stiffness": Muscle_Stiffness,
            "Obesity": Obesity
        }

        row = []
        for feat in self.X.columns.tolist():
            val = inputs.get(feat, 0)
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("yes", "y", "true"):
                    val_i = 1
                elif v in ("no", "n", "false"):
                    val_i = 0
                elif v in ("male", "m"):
                    val_i = 1
                elif v in ("female", "f"):
                    val_i = 0
                else:
                    try:
                        val_i = int(float(v))
                    except Exception:
                        val_i = 0
                row.append(val_i)
            else:
                try:
                    row.append(int(val))
                except Exception:
                    row.append(0)

        df_in = pd.DataFrame([row], columns=self.X.columns)
        for c in df_in.columns:
            if c == "Age":
                df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(0).astype(int)
            else:
                df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(0).astype(int)

        pred = self.tree.predict(df_in)
        return int(pred[0])