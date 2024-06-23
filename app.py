import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import stats

# Function to fetch data from SQLite
def fetch_data():
    conn = sqlite3.connect('predict.db')
    df = pd.read_sql_query("SELECT * FROM heart", conn)
    conn.close()
    return df

# Load dataset
df = fetch_data()

# Remove any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .title {
        font-size: 40px;
        color: #4A90E2;
    }
    .subheader {
        font-size: 20px;
        color: #4A90E2;
        margin-top: 20px;
    }
    .form {
        margin-top: 20px;
    }
    .input-row {
        margin-bottom: 10px;
    }
    .button {
        background-color: #4A90E2;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .footer {
        font-size: 12px;
        color: #777;
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f5f5f5;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Heart Disease Prediction App</h1>', unsafe_allow_html=True)

# Sidebar menu
menu = ["Home", "Explore", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown('<h2 class="subheader">Home</h2>', unsafe_allow_html=True)
    st.write("Welcome to the Heart Disease Prediction App. Use the sidebar to navigate through the app. You can explore the dataset or predict heart disease based on input features.")

elif choice == "Explore":
    st.markdown('<h2 class="subheader">Explore Data</h2>', unsafe_allow_html=True)

    st.write("### Target Variable Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    total = float(len(df))
    sns.countplot(x='target', data=df, hue='target', palette=['#4A90E2', '#FF0000'], ax=ax1, legend=False)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Heart Disease', 'Heart Disease'])
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.1f} %'.format((height / total) * 100), ha="center",
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
    sns.despine()
    st.pyplot(fig1)

    st.write("### Distribution of Numerical Features")
    num_feats = ['sex', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca']
    ncol = 3
    nrow = 2
    fig2, axes = plt.subplots(nrow, ncol, figsize=(25, 12))
    fig2.subplots_adjust(top=0.92, hspace=0.4)

    for i, col in enumerate(num_feats):
        row = i // ncol
        col_index = i % ncol
        ax = axes[row, col_index]
        if col == 'sex' or col == 'ca':
            sns.countplot(data=df, x=col, hue="target", palette=['#4A90E2', '#FF0000'], ax=ax)
            ax.set_ylabel("Count", fontsize=12)
        else:
            sns.kdeplot(data=df, x=col, hue="target", multiple="stack", palette=['#4A90E2', '#FF0000'], ax=ax)
            ax.set_ylabel("Density", fontsize=12)
        ax.set_xlabel(col.capitalize(), fontsize=12)
        sns.despine()
        if col == 'ca':
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.0f}'.format((height)), ha="center",
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    st.pyplot(fig2)

    st.write("### Additional Graphs")

    # Additional Graph 1: Reg plots of selected features
    fig3, ax3 = plt.subplots(1, 4, figsize=(20, 4))
    mypal = sns.color_palette("husl", 8)
    sns.regplot(data=df[df['target'] == 1], x='age', y='chol', ax=ax3[0], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='chol', ax=ax3[0], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='thalach', ax=ax3[1], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='thalach', ax=ax3[1], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='trestbps', ax=ax3[2], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='trestbps', ax=ax3[2], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='oldpeak', ax=ax3[3], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='oldpeak', ax=ax3[3], color=mypal[5], label='0')
    plt.suptitle('Reg plots of selected features')
    plt.legend()
    st.pyplot(fig3)

    # Additional Graph 2: Pairplot of numerical features
    num_feats = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca', 'target']
    data_pairplot = df[num_feats]
    g = sns.pairplot(data_pairplot, hue="target", corner=True, diag_kind='hist', palette=mypal[1::4])
    plt.suptitle('Pairplot: Numerical Features', fontsize=24)
    st.pyplot(g)

    # Additional Graph 3: Count plots of categorical features
    cat_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    fig4, axes4 = plt.subplots(2, 4, figsize=(20, 10))
    axes4 = axes4.flatten()
    for i, col in enumerate(cat_feats):
        sns.countplot(x=col, hue='target', data=df, palette=['#4A90E2', '#FF0000'], ax=axes4[i])
    plt.tight_layout()
    plt.suptitle('Count plots of categorical features', y=1.02)
    st.pyplot(fig4)

    # Additional Graph 4: Numerical features correlation heatmap (Pearson's)
    df_corr = df[num_feats].corr()
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    ax5.set_title("Numerical features correlation (Pearson's)", fontsize=20)
    st.pyplot(fig5)

    # Additional Graph 5: Point-biserial correlation heatmap
    def point_biserial(x, y):
        pb = stats.pointbiserialr(x, y)
        return pb[0]

    pb_matrix = np.zeros((len(num_feats), len(num_feats)))
    for i in range(len(num_feats)):
        for j in range(len(num_feats)):
            pb_matrix[i, j] = point_biserial(df[num_feats[i]], df[num_feats[j]])

    pb_df = pd.DataFrame(pb_matrix, index=num_feats, columns=num_feats)
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pb_df, annot=True, cmap='coolwarm',
    fmt=".2f", linewidths=.5)
    ax6.set_title("Point-biserial correlation heatmap", fontsize=20)
    st.pyplot(fig6)

    # Additional Graph 6: Categorical features correlation heatmap (Cramer's V)
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    cramers_matrix = np.zeros((len(cat_feats), len(cat_feats)))
    for i in range(len(cat_feats)):
        for j in range(len(cat_feats)):
            cramers_matrix[i, j] = cramers_v(df[cat_feats[i]], df[cat_feats[j]])

    cramers_df = pd.DataFrame(cramers_matrix, index=cat_feats, columns=cat_feats)
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cramers_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    ax7.set_title("Categorical features correlation (Cramer's V)", fontsize=20)
    st.pyplot(fig7)

elif choice == "Predict":
    st.markdown('<h2 class="subheader">Predict Heart Disease</h2>', unsafe_allow_html=True)
    st.write("Please input the following features to get a prediction:")

    age = st.number_input("Age (in years)", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

    # Mapping categorical values to numerical
    sex_mapping = {"Female": 0, "Male": 1}
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    fbs_mapping = {"False": 0, "True": 1}
    restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_mapping = {"No": 0, "Yes": 1}
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_mapping = {"Normal": 2, "Fixed Defect": 1, "Reversable Defect": 3}

    # Transform user inputs to numerical values
    sex = sex_mapping[sex]
    cp = cp_mapping[cp]
    fbs = fbs_mapping[fbs]
    restecg = restecg_mapping[restecg]
    exang = exang_mapping[exang]
    slope = slope_mapping[slope]
    thal = thal_mapping[thal]

    # Prepare input features for prediction
    input_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    input_features_scaled = scaler.transform(input_features)

    if st.button("Predict"):
        prediction = svm_model.predict(input_features_scaled)
        result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        st.success(f"The model predicts: {result}")

# Close the main div
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed by Mashaba Hlulani Charles</div>', unsafe_allow_html=True)
