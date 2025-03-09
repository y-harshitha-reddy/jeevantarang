import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from io import BytesIO
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tempfile
from folium import plugins
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report



# Load dataset from Excel file
def load_data(file_path):
    data = pd.read_excel(r"Endangered_Species_Dataset.xlsx")
    return data

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = ['Region', 'Habitat_Type']

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    numerical_columns = [
        'Current_Population', 'Population_Decline_Rate (%)', 'Average_Temperature (°C)',
        'Air_Quality_Index', 'Noise_Level (dB)', 'Protected_Areas (%)',
        'Migration_Distance (km)', 'Climate_Change_Risk (%)', 'Fragmentation_Risk (%)'
    ]
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # Ensure 'Extinction_Risk (%)' is available
    if 'Extinction_Risk (%)' in data.columns:
        bins = [0, 33, 66, 100]
        labels = ['Low', 'Medium', 'High']
        data['Extinction_Risk_Class'] = pd.cut(data['Extinction_Risk (%)'], bins=bins, labels=labels)
        
        # Convert categories to numerical labels
        le_risk = LabelEncoder()
        data['Extinction_Risk_Class'] = le_risk.fit_transform(data['Extinction_Risk_Class'])
    else:
        raise ValueError("Column 'Extinction_Risk (%)' not found in dataset!")

    X = data.drop(columns=['Species', 'Extinction_Risk (%)', 'Image_URL', 'Region', 'Habitat_Type'], errors='ignore')
    y = data['Extinction_Risk_Class']

    return X, y, scaler, label_encoders, numerical_columns, categorical_columns


# Geospatial visualization with image URL
def plot_species_on_map(data):
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    for _, row in data.iterrows():
        # Check if 'Image_URL' exists for the species
        image_url = row.get("Image_URL", None)
        popup_content = f"<b>{row['Species']}</b><br>Extinction Risk: {row['Extinction_Risk (%)']}%<br>Region: {row['Region']}<br>Habitat: {row['Habitat_Type']}"
        
        # If there's an image URL, add it to the popup
        if image_url:
            popup_content += f"<br><img src='{image_url}' width='250px' height='150px'>"
            
            # Create a custom DivIcon to use the image as the marker
            icon = folium.DivIcon(
                icon_size=(40, 40),  # Size of the icon
                icon_anchor=(20, 40),  # Position of the icon relative to the marker
                html=f'<img src="{image_url}" width="30" height="30" style="border-radius: 50%;">'
            )

            # Add the marker with the image as the icon
            folium.Marker(
                location=[row.get("Latitude", 0), row.get("Longitude", 0)],
                popup=folium.Popup(popup_content, max_width=300),
                icon=icon  # Use the custom DivIcon with the image
            ).add_to(m)

    st_folium(m, width=800, height=500)

def generate_report_with_graph(data):
    # Generate a bar plot for Species vs Extinction Risk
    plt.figure(figsize=(10, 6))
    plt.bar(data['Species'], data['Extinction_Risk (%)'], color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Extinction Risk (%)')
    plt.title('Extinction Risk by Species')
    plt.xticks(rotation=90)

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(tmpfile.name, format='png')
        plot_path = tmpfile.name
    plt.close()

    # Create the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add a title
    pdf.cell(200, 10, txt="Species Extinction Risk Report", ln=True, align='C')

    # Add species details and extinction risk
    pdf.set_font("Arial", size=10)
    for _, row in data.iterrows():
        pdf.cell(200, 10, txt=f"Species: {row['Species']}", ln=True)
        pdf.cell(200, 10, txt=f"Extinction Risk: {row['Extinction_Risk (%)']}%", ln=True)
        pdf.cell(200, 5, txt="", ln=True)  # Blank line for spacing

    # Add the graph to the PDF
    pdf.add_page()
    pdf.cell(200, 10, txt="Graph: Extinction Risk by Species", ln=True, align='C')
    pdf.image(plot_path, x=10, y=30, w=180)  # Adjust x, y, and width as needed

    # Save PDF to a buffer for download
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')  # Capture the output as a string
    pdf_buffer.write(pdf_output)  # Write to the buffer
    pdf_buffer.seek(0)  # Go to the beginning of the buffer

    # Provide the buffer as a downloadable file in Streamlit
    st.sidebar.download_button("Download PDF Report", pdf_buffer, file_name="species_report.pdf", mime="application/pdf")

# Visualizations for parameter comparison
def parameter_comparison(data, parameter):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Species", y=parameter, data=data)
    plt.title(f'Comparison of {parameter} Across Species')
    plt.xticks(rotation=90)
    st.pyplot(plt)

import numpy as np
import streamlit as st

def compute_information_gain(clf, X_train, y_train):
    """
    Function to compute Information Gain (IG) for each feature and provide insights.
    """

    # Function to calculate entropy
    def entropy(y):
        unique, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return -np.sum(prob * np.log2(prob + 1e-9))  # Small epsilon to avoid log(0)

    # Compute parent entropy (before any split)
    parent_entropy = entropy(y_train)
    IG_values = {}

    # Loop through each feature in the dataset
    for feature_idx in range(X_train.shape[1]):
        feature_values = X_train.iloc[:, feature_idx]
        split_value = np.median(feature_values)  # Using median as a split threshold
        left_mask = feature_values <= split_value
        right_mask = feature_values > split_value
        
        # Skip if the split is invalid (i.e., all values go to one side)
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue  

        # Compute entropy for left and right splits
        left_entropy = entropy(y_train[left_mask])
        right_entropy = entropy(y_train[right_mask])
        
        # Weighted entropy for the split
        N = len(y_train)
        N_left, N_right = left_mask.sum(), right_mask.sum()
        
        IG = parent_entropy - (N_left / N) * left_entropy - (N_right / N) * right_entropy
        IG_values[X_train.columns[feature_idx]] = IG

    # Sort features by highest Information Gain
    IG_values = dict(sorted(IG_values.items(), key=lambda x: x[1], reverse=True))
    
    return IG_values

# 📌 Add this after clf.fit(X_train, y_train)




def main():
    st.set_page_config(page_title="Species Extinction Risk", layout="wide")
    st.title("JEEVANTARANG ⧖")
    
    file_path = r"C:\\Users\\dhoni\\Music\\jt2\\Endangered_Species_Dataset.xlsx"
    data = load_data(file_path)
    
    if data is not None:
        # Define features and target
        features = [
            "Current_Population", "Population_Decline_Rate (%)", "Average_Temperature (°C)",
            "Climate_Change_Risk (%)", "Fragmentation_Risk (%)"
        ]
        target = "Extinction_Risk (%)"

        st.sidebar.header("🌲 Random Forest Model")

        # Ensure features exist in the dataset
        if not all(col in data.columns for col in features + [target]):
            st.error("Missing necessary columns in the dataset. Ensure all specified features and the target column are present.")
            return

        rf_features = st.sidebar.multiselect("Select Features for Random Forest", features, default=features)

        

        if rf_features:
            # Preprocess the data (adjusted to match the expected arguments of the function)
            X, y, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)
            X = X[rf_features]

            try:
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the Random Forest model
                rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
                rf_clf.fit(X_train, y_train)

                # Make predictions
                y_pred_rf = rf_clf.predict(X_test)
                accuracy_rf = accuracy_score(y_test, y_pred_rf)
                precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
                recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)

                # Display performance metrics in the sidebar
                st.sidebar.subheader("📊 Random Forest Performance")
                st.sidebar.write(f"**Accuracy:** {accuracy_rf:.2f}")
                st.sidebar.write(f"**Precision:** {precision_rf:.2f}")
                st.sidebar.write(f"**Recall:** {recall_rf:.2f}")

                # Provide feedback based on accuracy
                if accuracy_rf > 0.8:
                    st.sidebar.success("🌟 High accuracy! The model makes strong predictions.")
                elif accuracy_rf > 0.6:
                    st.sidebar.warning("⚠️ Moderate accuracy. Consider tuning hyperparameters.")
                else:
                    st.sidebar.error("🔻 Low accuracy! Consider adding more data or improving feature selection.")

                # Display feature importance
                st.subheader("🔥 Random Forest Feature Importance")
                rf_feature_importance = pd.DataFrame({
                    "Feature": rf_features,
                    "Importance": rf_clf.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                # Plot feature importance
                plt.figure(figsize=(8, 5))
                sns.barplot(x="Importance", y="Feature", data=rf_feature_importance, palette="coolwarm")
                plt.title("Feature Importance in Random Forest")
                st.pyplot(plt)

            except ValueError as e:
                st.sidebar.error(f"Error in Random Forest model training: {e}")

   
    st.sidebar.header("🔍 Decision Tree Model")
    features = [
        "Current_Population", "Population_Decline_Rate (%)", "Average_Temperature (°C)",
        "Climate_Change_Risk (%)", "Fragmentation_Risk (%)"
    ]
    target = "Extinction_Risk (%)"
    
    selected_features = st.sidebar.multiselect("Select Features for Decision Tree", features, default=features)
    selected_target = target
    
    
    if selected_features and selected_target:
        X, y, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)
        X = X[selected_features]
    
        
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            IG_results = compute_information_gain(clf, X_train, y_train)

# Streamlit Sidebar Output with Insights
            st.sidebar.subheader("📊 Information Gain for Each Feature")
            for feature, ig in IG_results.items():
                st.sidebar.write(f"**{feature}:** {ig:.4f}")

# 📝 Display Insights in the Main Area
            st.subheader("🔍 Insights from Information Gain Analysis(Decision Tree)")
            if IG_results:
                best_feature = max(IG_results, key=IG_results.get)
                worst_feature = min(IG_results, key=IG_results.get)
                st.write(f"✅ The **most important feature** is **{best_feature}** with an IG of **{IG_results[best_feature]:.4f}**.")
                st.write(f"⚠️ The **least important feature** is **{worst_feature}** with an IG of **{IG_results[worst_feature]:.4f}**.")
                st.write("🔹 Features with **higher IG values** contribute more to the decision-making process.")
            else:
                st.write("⚠️ No valid Information Gain could be calculated. Check the dataset!")
            

            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            st.sidebar.subheader("📊 Decision Tree Performance")
            st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
            if accuracy > 0.8:
                st.sidebar.success("🔹 High accuracy! The model is making strong predictions.")
            elif accuracy > 0.6:
                st.sidebar.warning("⚠️ Moderate accuracy. Consider improving data quality or tuning the model.")
            else:
                st.sidebar.error("🔻 Low accuracy! The model might be overfitting or missing key features.")
            
            st.sidebar.write(f"**Precision:** {precision:.2f}")
            if precision > 0.8:
                st.sidebar.success("✅ High precision! Most predicted species are correct.")
            elif precision > 0.6:
                st.sidebar.warning("⚠️ Moderate precision. Some misclassifications exist.")
            else:
                st.sidebar.error("🔻 Low precision. The model is predicting incorrectly too often.")
            
            st.sidebar.write(f"**Recall:** {recall:.2f}")
            if recall > 0.8:
                st.sidebar.success("🟢 High recall! The model is detecting most species correctly.")
            elif recall > 0.6:
                st.sidebar.warning("⚠️ Moderate recall. Some species are being missed.")
            else:
                st.sidebar.error("🔻 Low recall. The model is failing to detect important cases.")
            
        except ValueError as e:
            st.sidebar.error(f"Error in model training: {e}")
             

    st.subheader("📌 Decision Tree Visualization")
# Plot the decision tree
    plt.figure(figsize=(12, 6))
    plot_tree(clf, feature_names=selected_features, class_names=["Low", "Medium", "High"], filled=True)
    st.pyplot(plt)
    # Plot feature importance
    st.subheader("💡 Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
# Bar plot for feature importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance in Decision Tree")
    st.pyplot(plt)
    
    
    # NAIVE BAYES MODEL
    st.sidebar.header("🤖 Naïve Bayes Model")

    # Feature Selection in Sidebar
    all_features = [
        "Current_Population", "Population_Decline_Rate (%)", "Average_Temperature (°C)",
        "Climate_Change_Risk (%)", "Fragmentation_Risk (%)"
    ]
    selected_features = st.sidebar.multiselect("📌 Select Features for Naïve Bayes", all_features, default=all_features)

    target_variable = "Extinction_Risk (%)"

    # Ensure target variable exists
    if target_variable not in data.columns:
        st.sidebar.error(f"❌ Target variable '{target_variable}' not found in dataset!")
    else:
        # Preprocess Data
        X, _, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)

        # Select chosen features
        X = X[selected_features]
        y_continuous = data[target_variable]  # Continuous target variable

        # Convert Continuous Target into Categorical Labels (Low, Medium, High)
        bins = [0, 20, 40, 100]  # Define risk levels
        labels = ["Low", "Medium", "High"]
        y = pd.cut(y_continuous, bins=bins, labels=labels)

        try:
            # Encode Target Labels (Convert Categories to Numeric Values)
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
            nb_clf = GaussianNB()
            nb_clf.fit(X_train, y_train)

            # Predictions
            y_pred_nb = nb_clf.predict(X_test)

            # Performance Metrics
            accuracy_nb = accuracy_score(y_test, y_pred_nb)
            precision_nb = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
            recall_nb = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)

            # Display Performance
            st.sidebar.subheader("📊 Naïve Bayes Performance")
            st.sidebar.write(f"**Accuracy:** {accuracy_nb:.2f}")
            st.sidebar.write(f"**Precision:** {precision_nb:.2f}")
            st.sidebar.write(f"**Recall:** {recall_nb:.2f}")

            # Insights
            st.subheader("🧠 Insights from Naïve Bayes Model")
            if accuracy_nb > 0.8:
                st.success("🌟 The model is highly accurate! This suggests a strong correlation between the chosen features and extinction risk.")
            elif accuracy_nb > 0.6:
                st.warning("⚠️ The model shows moderate accuracy. Consider refining features or using additional data.")
            else:
                st.error("🔻 Low accuracy detected! This might indicate a need for feature engineering or alternative models.")

            # Feature Importance (Variance of Features)
            st.subheader("📌 Feature Importance")

            feature_variances = np.var(X_train, axis=0)
            feature_importance = feature_variances / np.sum(feature_variances)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=feature_importance, y=X_train.columns, palette="viridis")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.title("Feature Importance in Naïve Bayes Model")
            st.pyplot(fig)

        except ValueError as e:
            st.sidebar.error(f"Error in Naïve Bayes model training: {e}")
    


    # LINEAR REGRESSION MODEL
    st.sidebar.header("📈 Linear Regression Model")
    dependent_variable = st.sidebar.selectbox("Select Dependent Variable (Y):", data.columns)
    independent_variables = st.sidebar.multiselect(
        "Select Independent Variables (X):", [col for col in data.columns if col != dependent_variable]
    )

    if dependent_variable and independent_variables:
        # Preprocess the data
        X, y, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)
        X = X[independent_variables]
        y = data[dependent_variable]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

        # Coefficients
        coefficients = pd.DataFrame({
            "Feature": independent_variables,
            "Coefficient": model.coef_
        })

        # Display metrics and insights
        st.sidebar.subheader("Model Insights")
        st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.sidebar.write(f"**R-squared:** {r2:.4f}")
        st.sidebar.write(f"**Adjusted R-squared:** {adj_r2:.4f}")

        if r2 > 0.7:
            st.sidebar.write("🟢: The model explains a significant portion of the variance in the dependent variable.")
        elif r2 > 0.4:
            st.sidebar.write("🟡: The model explains a moderate portion of the variance. Consider adding more features or refining the data.")
        else:
            st.sidebar.write("🔴: The model has low explanatory power. Significant improvements are needed.")

        if mse < 1:
            st.sidebar.write("🟢: The model's predictions are very close to the actual values on average.")
        else:
            st.sidebar.write("🔴: The model's predictions have a higher average error. Further tuning might help.")

        # Coefficients table
        st.subheader("Feature Coefficients")
        st.write(coefficients)

        # Visualizations
        st.subheader("Visualizations")
        if len(independent_variables) == 1:
            # Scatterplot for single variable
            plt.figure(figsize=(8, 6))
            plt.scatter(X_test[independent_variables[0]], y_test, color='blue', label='Actual')
            plt.plot(X_test[independent_variables[0]], y_pred, color='red', label='Predicted')
            plt.title(f"Regression Line for {independent_variables[0]}")
            plt.xlabel(independent_variables[0])
            plt.ylabel(dependent_variable)
            plt.legend()
            st.pyplot(plt)
        else:
            # Residual plot for multiple variables
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, color='purple')
            plt.axhline(y=0, color='red', linestyle='--')
            plt.title("Residual Plot")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            st.pyplot(plt)

    species = data["Species"].unique()
    selected_species = st.sidebar.selectbox("Select Species", species)

    if selected_species:
        # Filter data for the selected species
        selected_row = data[data["Species"] == selected_species].iloc[0]

        # Display species details with image beside it
        st.header(f"Species: {selected_row['Species']}")

        # Two-column layout
        col1, col2 = st.columns([1, 3])

        # Column 1 (for image)
        image_url = selected_row.get("Image_URL", None)
        if pd.notnull(image_url):
            col1.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="{image_url}" style="width: 250px; height: auto; border-radius: 15px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);"/>
                    <p style="font-size: 14px; font-style: italic;">Image of {selected_row['Species']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            col1.warning("No image available for this species.")

        # Column 2 (for information)
        col2.write(f"**Region**: {selected_row['Region']}")
        col2.write(f"**Habitat Type**: {selected_row['Habitat_Type']}")
        col2.write(f"**Current Population**: {selected_row['Current_Population']}")
        col2.write(f"**Population Decline Rate**: {selected_row['Population_Decline_Rate (%)']}%")
        col2.write(f"**Climate Change Risk**: {selected_row['Climate_Change_Risk (%)']}%")
        col2.write(f"**Fragmentation Risk**: {selected_row['Fragmentation_Risk (%)']}%")
        col2.write(f"**Extinction Risk**: {selected_row['Extinction_Risk (%)']}%")


        # Risk warnings
        if selected_row['Climate_Change_Risk (%)'] > 80 or selected_row['Fragmentation_Risk (%)'] > 70:
            st.warning("High extinction risk due to climate change or habitat fragmentation.")
        elif selected_row['Current_Population'] < 5000:
            st.warning("Critical risk due to low population levels.")
        else:
            st.success("Moderate risk, but monitoring is advised.")

    comparison_parameter = st.sidebar.selectbox(
        "Select Column for Comparison",
        [
            'Current_Population', 'Population_Decline_Rate (%)', 'Average_Temperature (°C)',
            'Air_Quality_Index', 'Noise_Level (dB)', 'Protected_Areas (%)', 'Migration_Distance (km)',
            'Climate_Change_Risk (%)', 'Fragmentation_Risk (%)', 'Extinction_Risk (%)'
        ]
    )

    if comparison_parameter:
        st.header(f"Comparing {comparison_parameter} with Species")
        parameter_comparison(data, comparison_parameter)

    st.header("Geospatial Visualization")
    plot_species_on_map(data)

    generate_report_with_graph(data)

if __name__ == "__main__":
    main()



    
