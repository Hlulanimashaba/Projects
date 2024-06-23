from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import sqlite3
import streamlit as st
from scipy import stats

# Fetch data function remains the same
def fetch_data():
    connection = sqlite3.connect('predict.db')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM heart")
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    connection.close()
    return df

# Load the dataset from SQLite
df = fetch_data()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split features and target variable
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<h1 class="title">Heart Disease Prediction App</h1>', unsafe_allow_html=True)

menu = ["Home", "Explore", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown('<h2 class="subheader">Home</h2>', unsafe_allow_html=True)
    st.write("Welcome to the Heart Disease Prediction App. Use the sidebar to navigate through the app. You can explore the dataset or predict heart disease based on input features.")

elif choice == "Explore":
    st.markdown('<h2 class="subheader">Explore Data</h2>', unsafe_allow_html=True)
    df = fetch_data()

    st.write("### Target Variable Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='#F6F5F4')
    total = float(len(df))
    sns.countplot(x='target', data=df, hue='target', palette=['#4A90E2', '#FF0000'], ax=ax1, legend=False)
    ax1.set_facecolor('#F6F5F4')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Heart Disease', 'Heart Disease'])
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.1f} %'.format((height / total) * 100), ha="center",
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
    sns.despine(right=True)
    sns.despine(offset=5, trim=True)
    st.pyplot(fig1)

    st.write("### Distribution of Numerical Features")
    num_feats = ['sex', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca']
    ncol = 3
    nrow = 2
    fig2, axes = plt.subplots(nrow, ncol, figsize=(25, 12), facecolor='#F6F5F4')
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
        sns.despine(right=True)
        sns.despine(offset=0, trim=False)
        if col == 'ca':
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.0f}'.format((height)), ha="center",
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    st.pyplot(fig2)

    st.write("### Additional Graphs")

    # Additional Graph 1: Reg plots of selected features
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    mypal = sns.color_palette("husl", 8)
    sns.regplot(data=df[df['target'] == 1], x='age', y='chol', ax=ax[0], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='chol', ax=ax[0], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='thalach', ax=ax[1], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='thalach', ax=ax[1], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='trestbps', ax=ax[2], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='trestbps', ax=ax[2], color=mypal[5], label='0')
    sns.regplot(data=df[df['target'] == 1], x='age', y='oldpeak', ax=ax[3], color=mypal[0], label='1')
    sns.regplot(data=df[df['target'] == 0], x='age', y='oldpeak', ax=ax[3], color=mypal[5], label='0')
    plt.suptitle('Reg plots of selected features')
    plt.legend()
    st.pyplot(fig)

    # Additional Graph 2: Pairplot of numerical features
    num_feats = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca', 'target']
    data_ = df[num_feats]
    g = sns.pairplot(data_, hue="target", corner=True, diag_kind='hist', palette=mypal[1::4])
    plt.suptitle('Pairplot: Numerical Features', fontsize=24)
    # st.pyplot(g) # Removed to resolve the warning

    # Additional Graph 3: Count plots of categorical features
    cat_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    def count_plot(data, cat_feats):    
        L = len(cat_feats)
        ncol= 2
        nrow= int(np.ceil(L/ncol))
        remove_last= (nrow * ncol) - L

        fig, ax = plt.subplots(nrow, ncol, figsize=(18, 24), facecolor='#F6F5F4')    
        fig.subplots_adjust(top=0.92)
        ax.flat[-1].set_visible(False) 
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
        i = 0
        for row in range(nrow):
            for col in range(ncol):
                if i == L:
                    break
                sns.countplot(data=data, x=cat_feats[i], hue='target', palette=['#4A90E2', '#FF0000'], ax=ax[row, col])
                ax[row, col].set_title(cat_feats[i].capitalize(), fontsize=14)
                ax[row, col].set_ylabel("Count", fontsize=12)
                ax[row, col].set_xlabel("")
                sns.despine(right=True)
                sns.despine(offset=0, trim=False)
                i += 1
        plt.suptitle('Distribution of Categorical Features', fontsize=24)
        st.pyplot(fig)

    count_plot(df, cat_feats)

    # Additional Graph 4: Numerical features correlation heatmap (Pearson's)
    df_ = df[num_feats]
    corr = df_.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(mypal, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, annot=True, square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Numerical features correlation (Pearson's)", fontsize=20, y=1.05)
    st.pyplot(f)

    # Additional Graph 5: Point-biserial correlation heatmap
    feats_ = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca', 'target']
    def point_biserial(x, y):
        pb = stats.pointbiserialr(x, y)
        return pb[0]

    rows = []
    for x in feats_:
        col = []
        for y in feats_:
            pbs = point_biserial(df[x], df[y])
            col.append(round(pbs, 2))
        rows.append(col)

    pbs_results = np.array(rows)
    DF = pd.DataFrame(pbs_results, columns=df[feats_].columns, index=df[feats_].columns)
    mask = np.triu(np.ones_like(DF, dtype=bool))
    corr = DF.mask(mask)

    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(mypal, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1, center=0, annot=True, square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Cont feats vs target correlation (point-biserial)", fontsize=20, y=1.05)
    st.pyplot(f)

    # Additional Graph 6: Categorical features correlation heatmap (Cramer's V)
    def cramers_v(x, y): 
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1) * (r-1)) / (n-1))
        rcorr = r - ((r-1)**2) / (n-1)
        kcorr = k - ((k-1)**2) / (n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    data_ = df[cat_feats]
    rows = []
    for x in data_:
        col = []
        for y in data_:
            cramers = cramers_v(data_[x], data_[y])
            col.append(round(cramers, 2))
        rows.append(col)

    cramers_results = np.array(rows)
    df_ = pd.DataFrame(cramers_results, columns=data_.columns, index=data_.columns)

    mypal_1 = ['#FC05FB', '#FEAEFE', '#FCD2FC', '#F3FEFA', '#B4FFE4', '#3FFEBA', '#FC05FB', '#FEAEFE', '#FCD2FC']
    mask = np.triu(np.ones_like(df_, dtype=bool))
    corr = df_.mask(mask)
    f, ax = plt.subplots(figsize=(10, 6), facecolor=None)
    cmap = sns.color_palette(mypal_1, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0, annot=True, square=False, linewidths=.01, cbar_kws={"shrink": 0.75})
    ax.set_title("Categorical Features Correlation (Cramer's V)", fontsize=20, y=1.05)
    st.pyplot(f)

elif choice == "Predict":
    st.markdown('<h2 class="subheader">Predict Heart Disease</h2>', unsafe_allow_html=True)
    st.write("Please input the following features to get a prediction:")

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    input_features = [age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    input_features_df = pd.DataFrame([input_features], columns=X.columns)
    input_features_scaled = scaler.transform(input_features_df)

    if st.button("Predict"):
        prediction = svm_model.predict(input_features_scaled)
        result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        st.success(f"The model predicts: {result}")

st.markdown('</div>', unsafe_allow_html=True)
