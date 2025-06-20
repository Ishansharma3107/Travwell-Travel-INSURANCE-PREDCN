import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components
import pickle

# --- Function to render notebook as HTML ---
def render_notebook_html(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.exclude_input_prompt = True
    html_exporter.exclude_output_prompt = True
    html_exporter.template_name = 'lab'  # clean, modern style
    (body, _) = html_exporter.from_notebook_node(notebook)
    components.html(body, height=1200, scrolling=True)

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Insurance Prediction",
        page_icon="💸",
        initial_sidebar_state="expanded",
    )

    st.title('💸 Travel Insurance Prediction')
    st.image('img.jpg', width=700)

    st.markdown(
        """
        In this project, I tackled the task of **Insurance Purchase Prediction** using Machine Learning in Python.
        The dataset, sourced from Kaggle, contains information about previous customers of a travel insurance company.
        The objective is to build and train a machine learning model that can predict whether an individual is likely
        to purchase travel insurance based on their demographic and travel-related attributes.
        """
    )

st.subheader("📓 Notebook (Code + Output)")
render_notebook_html("Travel_insurance.ipynb")

st.subheader("📝 Manual Prediction Input")
age = st.number_input('Age', min_value=0, step=1, help='Enter age in years')
option = st.selectbox('Employment Type', ('Government Sector', 'Private Sector/Self Employed'))
employmentType = 0 if option == 'Government Sector' else 1
annualIncome = st.number_input('Annual Income', min_value=0, step=1, help='Annual income in USD or local currency')
familyMembers = st.number_input('Family Members', min_value=1, step=1, help='Number of family members')
graduateOrNot = st.checkbox('Graduated')
chronicDiseases = st.checkbox('Chronic Diseases')
frequentFlyer = st.checkbox('Frequent Flyer')
everTravelledAbroad = st.checkbox('Ever Travelled Abroad')

features = [age, employmentType, int(graduateOrNot), annualIncome, familyMembers, int(chronicDiseases), int(frequentFlyer),
                int(everTravelledAbroad)]

# Load model and scaler
model = pickle.load(open('svc_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

if st.button('PREDICT'):
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        if prediction == 0:
            st.success('✅ Will Buy Insurance')
        else:
            st.error('❌ Will Not Buy Insurance')

if __name__ == "__main__":
    main()
