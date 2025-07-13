import streamlit as st
import pandas as pd
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os


st.set_page_config(
    page_title="Methane Emission Dashboard", 
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%;}
    .stSelectbox, .stSlider {margin-bottom: 20px;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    .warning-card {background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .success-card {background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .danger-card {background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .optimization-box {border: 1px solid #dee2e6; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)

# Breed number to name mapping (including Jersey)
BREED_MAPPING = {
    0: "Holstein",
    1: "Angus",
    2: "Hereford",
    3: "Simmental",
    4: "Jersey",  # Added Jersey breed
    5: "Other"   # For any unknown breeds
}

# Create reverse mapping for selection
BREED_OPTIONS = {name: num for num, name in BREED_MAPPING.items()}

# Breed-specific recommendations
BREED_TIPS = {
    "Holstein": "Holsteins typically benefit from higher fiber diets",
    "Angus": "Angus cattle often require balanced protein levels",
    "Hereford": "Herefords may need adjusted grazing hours",
    "Simmental": "Simmentals often respond well to varied diets",
    "Jersey": "Jerseys typically benefit from higher fat content",
    "Other": "Monitor this breed closely for optimal results"
}

@st.cache_resource
def load_model():
    """Load the trained LightGBM model with error handling"""
    try:
        model_path = 'models/methane_emission_model.txt'
        if os.path.exists(model_path):
            return lgb.Booster(model_file=model_path)
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    """Load and preprocess the dataset with fallback to simulated data"""
    try:
        data = pd.read_csv('data/processed_methane_data.csv')
        data.columns = data.columns.str.replace(' ', '_')
        
        if "Methane_Emissions" not in data.columns:
            data["Methane_Emissions"] = np.random.uniform(100, 400, len(data))
            
        return data
    
    except Exception as e:
        st.warning(f"Using simulated data due to: {str(e)}")
        data = pd.DataFrame({
            "Cattle_Breed": np.random.choice(list(BREED_MAPPING.values())[:-1], 100),  # Exclude "Other"
            "Age_Years": np.random.randint(1, 15, 100),
            "Weight_kg": np.random.randint(300, 800, 100),
            "Fiber_Content_%": np.random.uniform(10, 30, 100),
            "Protein_Content_%": np.random.uniform(12, 20, 100),
            "Carbohydrates_Content_%": np.random.uniform(30, 50, 100),
            "Fat_Content_%": np.random.uniform(3, 10, 100),
            "Grazing_Hours_per_Day": np.random.randint(4, 12, 100),
            "Activity_Level": np.random.choice(['Low', 'Medium', 'High'], 100),
            "Methane_Emissions": np.random.uniform(100, 400, 100)
        })
        
        # Enhance breed differences in synthetic data
        breed_factors = {
            "Holstein": 1.4,  # High emitter
            "Angus": 1.1,
            "Hereford": 1.0,   # Baseline
            "Simmental": 0.8,  # Low emitter
            "Jersey": 1.2      # Medium-high emitter
        }
        data["Methane_Emissions"] *= data["Cattle_Breed"].map(breed_factors)
        
        return data

# Load data and model
model = load_model()
data = load_data()

# Get required features from model
try:
    required_features = model.feature_name() if hasattr(model, 'feature_name') else [
        col for col in data.columns if col != "Methane_Emissions"
    ]
except Exception as e:
    st.error(f"Could not determine required features: {str(e)}")
    st.stop()

# Initialize session state
if 'input_df' not in st.session_state:
    st.session_state.input_df = pd.DataFrame()

# Sidebar - Input Parameters
st.sidebar.header("🛠️ Input Parameters")

# Get available breeds (convert numbers to names if needed)
if 'Cattle_Breed' in data.columns:
    if data['Cattle_Breed'].dtype in [np.int64, np.float64]:
        available_breeds = sorted(data['Cattle_Breed'].map(BREED_MAPPING).dropna().unique())
    else:
        available_breeds = sorted(data['Cattle_Breed'].unique())
else:
    available_breeds = list(BREED_MAPPING.values())

# Ensure all standard breeds are available
for breed in BREED_MAPPING.values():
    if breed not in available_breeds and breed != "Other":
        available_breeds.append(breed)

# Breed selection with names
selected_breed_name = st.sidebar.selectbox(
    'Cattle Breed',
    options=sorted(available_breeds),
    index=0,
    help="Select the primary breed of cattle"
)

# Show breed-number mapping to user
with st.sidebar.expander("🔢 Breed Number Reference"):
    st.markdown("**Breed codes used in the model:**")
    breed_df = pd.DataFrame.from_dict(BREED_MAPPING, orient='index', columns=['Breed Name'])
    st.table(breed_df.reset_index().rename(columns={'index': 'Code'}))

# Get numerical value for model
try:
    selected_breed_num = BREED_OPTIONS[selected_breed_name]
except KeyError:
    st.sidebar.warning(f"Breed '{selected_breed_name}' not in standard mapping, using 'Other'")
    selected_breed_num = BREED_OPTIONS["Other"]

# Show breed-specific tip
if selected_breed_name in BREED_TIPS:
    st.sidebar.info(f"ℹ️ {BREED_TIPS[selected_breed_name]}")

# Input controls with improved ranges and defaults
col1, col2 = st.sidebar.columns(2)

with col1:
    age_years = st.slider(
        'Age (Years)', 
        min_value=1, max_value=15, value=5,
        help="Age of the cattle in years"
    )
    
    weight_kg = st.slider(
        'Weight (kg)', 
        min_value=300, max_value=1200, value=600,
        help="Current weight of the cattle"
    )
    
    fiber_content = st.slider(
        'Fiber Content (%)', 
        min_value=5.0, max_value=50.0, value=20.0, step=0.5,
        help="Percentage of fiber in diet"
    )
    
    protein_content = st.slider(
        'Protein Content (%)', 
        min_value=5.0, max_value=30.0, value=16.0, step=0.5,
        help="Percentage of protein in diet"
    )

with col2:
    fat_content = st.slider(
        'Fat Content (%)', 
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="Percentage of fat in diet"
    )
    
    carb_content = st.slider(
        'Carbohydrates (%)', 
        min_value=20.0, max_value=60.0, value=40.0, step=0.5,
        help="Percentage of carbohydrates in diet"
    )
    
    grazing_hours = st.slider(
        'Grazing Hours', 
        min_value=0, max_value=24, value=8,
        help="Hours spent grazing per day"
    )
    
    activity_level = st.selectbox(
        'Activity Level', 
        ['Low', 'Medium', 'High'],
        help="Daily activity level of the cattle"
    )

# Prepare input data
input_data = {
    'Cattle_Breed': [selected_breed_num],  # Using numerical value
    'Age_Years': [age_years],
    'Weight_kg': [weight_kg],
    'Fiber_Content_%': [fiber_content],
    'Protein_Content_%': [protein_content],
    'Carbohydrates_Content_%': [carb_content],
    'Fat_Content_%': [fat_content],
    'Grazing_Hours_per_Day': [grazing_hours],
    'Activity_Level': [activity_level],
}

input_df = pd.DataFrame(input_data)
st.session_state.input_df = input_df.copy()

def safe_encode(encoder, value, classes):
    """Handle unseen categories during encoding"""
    if value in classes:
        return encoder.transform([value])[0]
    return len(classes)  # Assign to "unknown" category

# Encode categorical variables (only for activity level now)
try:
    activity_classes = data['Activity_Level'].astype(str).unique()
    activity_encoder = LabelEncoder().fit(activity_classes)
    input_df['Activity_Level'] = input_df['Activity_Level'].apply(
        lambda x: safe_encode(activity_encoder, str(x), activity_classes))
except Exception as e:
    st.error(f"Error encoding features: {str(e)}")
    st.stop()

# Ensure all required features are present
for feature in required_features:
    if feature not in input_df.columns:
        default_value = data[feature].median() if feature in data.columns else 0
        input_df[feature] = default_value

input_df = input_df[required_features]

# Main Dashboard Layout
st.title("🐄 Methane Emission Prediction & Analysis Dashboard")
st.markdown("Predict and analyze methane emissions from cattle based on various characteristics and diet composition.")

# Prediction Section
if st.button('🔍 Calculate Methane Emission', type="primary"):
    with st.spinner('Making prediction...'):
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Determine emission status
            if prediction < 120:
                status = "🟢 Low"
                status_class = "success-card"
            elif prediction < 150:
                status = "🟡 Moderate"
                status_class = "warning-card"
            else:
                status = "🔴 High"
                status_class = "danger-card"
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Emission", f"{prediction:.1f} g/day", help="Daily methane emission prediction")
            
            with col2:
                st.metric("Emission Status", status, help="Compared to typical cattle emissions")
            
            with col3:
                efficiency = (1 - (prediction / 400)) * 100  # 400g as worst case
                st.metric("Feed Efficiency", f"{max(0, efficiency):.1f}%", help="Higher is better")
            
            # Health recommendations
            st.subheader("📢 Health Recommendations")
            
            if prediction >150:
                st.markdown(f"""
                <div class="{status_class}">
                **⚠️ High Emission Alert**  
                - Reduce fiber content by 10-15%
                - Increase grazing time by 1-2 hours
                - Consider probiotics supplementation
                - Monitor weight and adjust feed accordingly
                </div>
                """, unsafe_allow_html=True)
            elif prediction > 120:
                st.markdown(f"""
                <div class="{status_class}">
                **🟡 Moderate Emission Levels**  
                - Optimal range for most cattle
                - Maintain current diet with minor variations
                - Regular weight monitoring recommended
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="{status_class}">
                **🟢 Low Emission Levels**  
                - Excellent feed efficiency
                - Current diet is well-balanced
                - Continue current management practices
                </div>
                """, unsafe_allow_html=True)
            
            # Diet Composition Analysis
            st.subheader("🍽️ Diet Composition Analysis")
            
            # Get current diet percentages
            diet_data = {
                "Component": ["Fiber", "Protein", "Carbs", "Fat"],
                "Current": [
                    input_df["Fiber_Content_%"].iloc[0],
                    input_df["Protein_Content_%"].iloc[0],
                    input_df["Carbohydrates_Content_%"].iloc[0],
                    input_df["Fat_Content_%"].iloc[0]
                ],
                "Optimal": [18, 15, 45, 5]  # Example optimal values
            }
            
            fig = px.line_polar(
                pd.DataFrame(diet_data),
                r="Current",
                theta="Component",
                line_close=True,
                title="Current vs Optimal Diet Composition",
                template="plotly_dark",
                markers=True
            )
            fig.add_trace(go.Scatterpolar(
                r=diet_data["Optimal"],
                theta=diet_data["Component"],
                name="Optimal",
                fill="toself",
                line=dict(color="green")
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretation**:  
            - Closer to green = Better emission control  
            - Large gaps indicate optimization opportunities
            """)
            
            # Emissions Hotspot Map
            st.subheader("🌍 Global Emission Hotspots")
            
            # Simulated geo data with realistic clusters
            def generate_hotspots():
                base_locations = {
                    "USA": (37.09, -95.71),
                    "Brazil": (-14.24, -51.93),
                    "India": (20.59, 78.96),
                    "Australia": (-25.27, 133.78),
                    "EU": (54.53, 15.26)
                }
                
                hotspots = []
                for region, (lat, lon) in base_locations.items():
                    for _ in range(10):
                        hotspots.append({
                            "Latitude": lat + np.random.uniform(-5, 5),
                            "Longitude": lon + np.random.uniform(-10, 10),
                            "Emissions": np.random.uniform(50, 400),
                            "Region": region,
                            "Farm": f"{region} Farm {np.random.randint(1,100)}"
                        })
                return pd.DataFrame(hotspots)
            
            geo_data = generate_hotspots()
            
            fig = px.scatter_mapbox(
                geo_data,
                lat="Latitude",
                lon="Longitude",
                color="Emissions",
                size="Emissions",
                hover_name="Farm",
                hover_data=["Region", "Emissions"],
                color_continuous_scale=px.colors.sequential.Redor,
                zoom=1,
                height=600
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                title="Methane Emission Hotspots by Region"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Diet Optimization Planner
            st.subheader("📅 Weekly Diet Optimization Planner")
            
            with st.expander("✏️ Set Optimization Goals"):
                target_emission = st.slider(
                    "Target Emission (g/day)",
                    min_value=50,
                    max_value=int(prediction),
                    value=int(prediction * 0.8),  # Default: 20% reduction
                    step=5
                )
                weeks = st.slider("Weeks to achieve target", 1, 12, 4)
            
            # Calculate weekly progression
            weekly_reduction = (prediction - target_emission) / weeks
            weekly_diet_changes = {
                "Fiber": -0.5 * weekly_reduction,
                "Protein": 0.3 * weekly_reduction,
                "Grazing": 0.2 * weekly_reduction
            }
            
            # Generate weekly plan
            plan = []
            current_emission = prediction
            current_fiber = input_df["Fiber_Content_%"].iloc[0]
            current_protein = input_df["Protein_Content_%"].iloc[0]
            current_grazing = input_df["Grazing_Hours_per_Day"].iloc[0]
            
            for week in range(1, weeks+1):
                current_emission -= weekly_reduction
                current_fiber += weekly_diet_changes["Fiber"]
                current_protein += weekly_diet_changes["Protein"]
                current_grazing += weekly_diet_changes["Grazing"]
                
                plan.append({
                    "Week": week,
                    "Target Emission": max(target_emission, current_emission),
                    "Fiber (%)": round(max(10, current_fiber), 1),
                    "Protein (%)": round(min(30, current_protein), 1),
                    "Grazing (hrs)": round(min(12, current_grazing), 1),
                    "Notes": f"Monitor weight weekly" if week % 2 == 0 else "Normal observation"
                })
            
            plan_df = pd.DataFrame(plan)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    plan_df.style.background_gradient(
                        subset=["Target Emission"],
                        cmap="RdYlGn_r"
                    ),
                    height=300,
                    use_container_width=True
                )
            
            with col2:
                fig = px.line(
                    plan_df,
                    x="Week",
                    y="Target Emission",
                    markers=True,
                    title="Weekly Emission Reduction Plan",
                    labels={"Target Emission": "Emission (g/day)"}
                )
                fig.update_layout(yaxis_range=[target_emission-10, prediction+10])
                st.plotly_chart(fig, use_container_width=True)
            
            st.download_button(
                label="📥 Download Diet Plan",
                data=plan_df.to_csv(index=False),
                file_name=f"diet_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Save scenario option
            if st.button("💾 Save This Plan"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                scenario_record = {
                    "timestamp": timestamp,
                    "breed": selected_breed_name,
                    "prediction": prediction,
                    "target": target_emission,
                    "weeks": weeks,
                    **input_data
                }
                
                st.session_state.setdefault('history', []).append(scenario_record)
                st.success("Optimization plan saved to session history")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Data Explorer Section
st.sidebar.markdown("---")
if st.sidebar.checkbox("📁 Show Data Explorer"):
    st.subheader("🔍 Data Explorer")
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        min_emission = st.slider(
            "Minimum emission",
            min_value=int(data['Methane_Emissions'].min()),
            max_value=int(data['Methane_Emissions'].max()),
            value=int(data['Methane_Emissions'].quantile(0.25)))
    
    with col2:
        max_emission = st.slider(
            "Maximum emission",
            min_value=int(data['Methane_Emissions'].min()),
            max_value=int(data['Methane_Emissions'].max()),
            value=int(data['Methane_Emissions'].quantile(0.75)))
    
    filtered_data = data[
        (data['Methane_Emissions'] >= min_emission) & 
        (data['Methane_Emissions'] <= max_emission)
    ]
    
    st.dataframe(
        filtered_data.style.background_gradient(
            subset=['Methane_Emissions'], 
            cmap='RdYlGn_r'
        ),
        height=300,
        use_container_width=True
    )
    
    # Download button
    st.download_button(
        label="📥 Download Filtered Data",
        data=filtered_data.to_csv(index=False),
        file_name="filtered_methane_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard**:  
This tool predicts methane emissions from cattle based on various factors including breed, diet, and activity levels.  
*Developed for sustainable livestock management*  
*Last updated: {}*  
""".format(datetime.now().strftime("%Y-%m-%d")))
