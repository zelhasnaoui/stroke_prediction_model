"""
Main application for stroke prediction system using KNN model
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

class StrokePredictionApp:
    """
    Main application class for stroke prediction using KNN model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_encoders = {}
        self.model_loaded = False
        
    def load_trained_model(self, model_path='models/best_model.pkl', 
                          scaler_path='models/scaler.pkl', 
                          features_path='models/selected_features.pkl'):
        """Load the trained KNN model and preprocessing objects"""
        try:
            # Check if model files exist
            if not all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
                print("Model files not found. Please run the Feature.ipynb notebook first to train the model.")
                return False
            
            # Load model and preprocessing objects
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.selected_features = joblib.load(features_path)
            
            print(f"✓ KNN model loaded successfully from {model_path}")
            print(f"✓ Scaler loaded from {scaler_path}")
            print(f"✓ Selected features loaded from {features_path}")
            print(f"✓ Model ready for predictions with {len(self.selected_features)} features")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _encode_categorical_features(self, patient_data):
        """Encode categorical features according to the training pipeline"""
        encoded_data = patient_data.copy()
        
        # Gender encoding (Male=1, Female=0)
        if 'gender' in encoded_data:
            encoded_data['gender'] = 1 if encoded_data['gender'].lower() == 'male' else 0
        
        # Ever married encoding (Yes=1, No=0)
        if 'ever_married' in encoded_data:
            encoded_data['ever_married'] = 1 if encoded_data['ever_married'].lower() == 'yes' else 0
        
        # Residence type encoding (Urban=1, Rural=0)
        if 'Residence_type' in encoded_data:
            encoded_data['Residence_type'] = 1 if encoded_data['Residence_type'].lower() == 'urban' else 0
        
        # Work type encoding
        if 'work_type' in encoded_data:
            work_type_mapping = {
                'private': 0,
                'self-employed': 1,
                'govt_job': 2,
                'children': 3,
                'never_worked': 4
            }
            encoded_data['work_type'] = work_type_mapping.get(encoded_data['work_type'].lower(), 0)
        
        # Smoking status encoding
        if 'smoking_status' in encoded_data:
            smoking_mapping = {
                'never smoked': 0,
                'formerly smoked': 1,
                'smokes': 2,
                'unknown': 3
            }
            encoded_data['smoking_status'] = smoking_mapping.get(encoded_data['smoking_status'].lower(), 0)
        
        return encoded_data
    
    def _create_derived_features(self, patient_data):
        """Create derived features as in the training pipeline"""
        features = patient_data.copy()
        
        # Age group encoding (already numerical from training)
        if 'age' in features:
            if features['age'] < 30:
                features['age_group'] = 0
            elif features['age'] < 45:
                features['age_group'] = 1
            elif features['age'] < 60:
                features['age_group'] = 2
            elif features['age'] < 75:
                features['age_group'] = 3
            else:
                features['age_group'] = 4
        
        # BMI category encoding (already numerical from training)
        if 'bmi' in features:
            if features['bmi'] < 18.5:
                features['bmi_category'] = 0  # Underweight
            elif features['bmi'] < 25:
                features['bmi_category'] = 1  # Normal
            elif features['bmi'] < 30:
                features['bmi_category'] = 2  # Overweight
            else:
                features['bmi_category'] = 3  # Obese
        
        # Glucose level transformation (multiply by 100)
        if 'avg_glucose_level' in features:
            features['avg_glucose_level_x100'] = int(features['avg_glucose_level'] * 100)
        
        # Glucose category (one-hot encoded in training, but we'll use numerical here)
        if 'avg_glucose_level_x100' in features:
            if features['avg_glucose_level_x100'] < 70:
                features['glucose_category'] = 0  # Low
            elif features['avg_glucose_level_x100'] < 100:
                features['glucose_category'] = 1  # Normal
            elif features['avg_glucose_level_x100'] < 125:
                features['glucose_category'] = 2  # High
            else:
                features['glucose_category'] = 3  # Very High
        
        # Risk score calculation
        risk_factors = 0
        if features.get('hypertension', 0) == 1:
            risk_factors += 1
        if features.get('heart_disease', 0) == 1:
            risk_factors += 1
        if features.get('smoking_status', 0) in [1, 2]:  # Formerly smoked or smokes
            risk_factors += 1
        if features.get('age_group', 0) >= 2:  # Age 60+
            risk_factors += 1
        if features.get('bmi_category', 0) >= 2:  # Overweight or obese
            risk_factors += 1
        
        features['risk_score'] = risk_factors
        
        # Interaction features
        if 'age_group' in features and 'hypertension' in features:
            features['age_hypertension'] = features['age_group'] * features['hypertension']
        
        if 'bmi_category' in features and 'avg_glucose_level_x100' in features:
            features['bmi_glucose'] = features['bmi_category'] * (features['avg_glucose_level_x100'] / 100)
        
        if 'smoking_status' in features and 'heart_disease' in features:
            features['smoking_heart'] = features['smoking_status'] * features['heart_disease']
        
        return features
    
    def predict_single_patient(self, patient_data):
        """Predict stroke risk for a single patient"""
        if not self.model_loaded:
            print("No model loaded. Please load the trained model first.")
            return None
        
        try:
            # Convert to DataFrame if it's a dictionary
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            else:
                df = patient_data.copy()
            
            # Encode categorical features
            df_encoded = self._encode_categorical_features(df.iloc[0]).to_frame().T
            
            # Create derived features
            df_features = self._create_derived_features(df_encoded.iloc[0]).to_frame().T
            
            # Select only the features used in training
            available_features = [col for col in self.selected_features if col in df_features.columns]
            missing_features = [col for col in self.selected_features if col not in df_features.columns]
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    df_features[feature] = 0
            
            # Select and order features as in training
            X = df_features[self.selected_features]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            return {
                'prediction': prediction,
                'probability_no_stroke': probabilities[0],
                'probability_stroke': probabilities[1],
                'risk_level': self._get_risk_level(probabilities[1]),
                'confidence': max(probabilities)
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def _get_risk_level(self, stroke_probability):
        """Determine risk level based on probability"""
        if stroke_probability < 0.2:
            return "Low"
        elif stroke_probability < 0.5:
            return "Moderate"
        elif stroke_probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def interactive_prediction(self):
        """Interactive prediction interface"""
        print("\n" + "="*60)
        print("STROKE RISK PREDICTION - INTERACTIVE MODE")
        print("="*60)
        print("Enter patient information for stroke risk assessment:")
        print("(Press Ctrl+C to cancel at any time)")
        
        try:
            # Collect patient data
            patient_data = {}
            
            print("\n📋 DEMOGRAPHIC INFORMATION")
            print("-" * 30)
            
            # Gender
            while True:
                gender = input("Gender (Male/Female): ").strip().lower()
                if gender in ['male', 'female']:
                    patient_data['gender'] = gender.title()
                    break
                print("Please enter 'Male' or 'Female'")
            
            # Age
            while True:
                try:
                    age = float(input("Age (years): "))
                    if 0 <= age <= 120:
                        patient_data['age'] = age
                        break
                    print("Please enter a valid age between 0 and 120")
                except ValueError:
                    print("Please enter a valid number")
            
            # Marital status
            while True:
                married = input("Married (Yes/No): ").strip().lower()
                if married in ['yes', 'no']:
                    patient_data['ever_married'] = married.title()
                    break
                print("Please enter 'Yes' or 'No'")
            
            # Work type
            print("Work type options: Private, Self-employed, Govt_job, Children, Never_worked")
            while True:
                work = input("Work type: ").strip().lower()
                valid_work = ['private', 'self-employed', 'govt_job', 'children', 'never_worked']
                if work in valid_work:
                    patient_data['work_type'] = work
                    break
                print("Please enter a valid work type from the options above")
            
            # Residence type
            while True:
                residence = input("Residence type (Urban/Rural): ").strip().lower()
                if residence in ['urban', 'rural']:
                    patient_data['Residence_type'] = residence.title()
                    break
                print("Please enter 'Urban' or 'Rural'")
            
            print("\n🏥 MEDICAL INFORMATION")
            print("-" * 30)
            
            # Hypertension
            while True:
                try:
                    hypertension = int(input("Hypertension (0=No, 1=Yes): "))
                    if hypertension in [0, 1]:
                        patient_data['hypertension'] = hypertension
                        break
                    print("Please enter 0 or 1")
                except ValueError:
                    print("Please enter 0 or 1")
            
            # Heart disease
            while True:
                try:
                    heart_disease = int(input("Heart disease (0=No, 1=Yes): "))
                    if heart_disease in [0, 1]:
                        patient_data['heart_disease'] = heart_disease
                        break
                    print("Please enter 0 or 1")
                except ValueError:
                    print("Please enter 0 or 1")
            
            # Average glucose level
            while True:
                try:
                    glucose = float(input("Average glucose level (mg/dL): "))
                    if 50 <= glucose <= 500:
                        patient_data['avg_glucose_level'] = glucose
                        break
                    print("Please enter a valid glucose level between 50 and 500 mg/dL")
                except ValueError:
                    print("Please enter a valid number")
            
            # BMI
            while True:
                try:
                    bmi = float(input("BMI (Body Mass Index): "))
                    if 10 <= bmi <= 60:
                        patient_data['bmi'] = bmi
                        break
                    print("Please enter a valid BMI between 10 and 60")
                except ValueError:
                    print("Please enter a valid number")
            
            # Smoking status
            print("Smoking status options: Never smoked, Formerly smoked, Smokes, Unknown")
            while True:
                smoking = input("Smoking status: ").strip().lower()
                valid_smoking = ['never smoked', 'formerly smoked', 'smokes', 'unknown']
                if smoking in valid_smoking:
                    patient_data['smoking_status'] = smoking
                    break
                print("Please enter a valid smoking status from the options above")
            
            # Make prediction
            print("\n🔄 PROCESSING PREDICTION...")
            print("-" * 30)
            
            result = self.predict_single_patient(patient_data)
            
            if result:
                self._display_results(result)
            else:
                print("❌ Error occurred during prediction. Please try again.")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Prediction cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    def _display_results(self, result):
        """Display prediction results in a formatted way"""
        print("\n" + "="*60)
        print("🎯 STROKE RISK ASSESSMENT RESULTS")
        print("="*60)
        
        # Main prediction
        prediction_text = "🚨 STROKE RISK DETECTED" if result['prediction'] == 1 else "✅ NO STROKE RISK"
        print(f"\n{prediction_text}")
        
        # Probabilities
        print(f"\n📊 PROBABILITY ANALYSIS:")
        print(f"   • No Stroke: {result['probability_no_stroke']:.1%}")
        print(f"   • Stroke Risk: {result['probability_stroke']:.1%}")
        print(f"   • Confidence: {result['confidence']:.1%}")
        
        # Risk level
        risk_emoji = {
            "Low": "🟢",
            "Moderate": "🟡", 
            "High": "🟠",
            "Very High": "🔴"
        }
        print(f"\n⚠️  RISK LEVEL: {risk_emoji.get(result['risk_level'], '❓')} {result['risk_level']}")
        
        # Recommendations
        self._print_recommendations(result['probability_stroke'])
        
        print("\n" + "="*60)
    
    def _print_recommendations(self, stroke_probability):
        """Print recommendations based on risk level"""
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        
        if stroke_probability < 0.2:
            print("🟢 LOW RISK - Continue maintaining a healthy lifestyle!")
            print("   • Regular health checkups")
            print("   • Maintain current healthy habits")
        elif stroke_probability < 0.5:
            print("🟡 MODERATE RISK - Consult a doctor for regular monitoring.")
            print("   • Schedule regular medical checkups")
            print("   • Monitor blood pressure and glucose levels")
            print("   • Consider lifestyle modifications")
        elif stroke_probability < 0.8:
            print("🟠 HIGH RISK - Urgent medical consultation recommended!")
            print("   • Schedule immediate medical consultation")
            print("   • Consider preventive medications")
            print("   • Implement strict lifestyle changes")
        else:
            print("🔴 VERY HIGH RISK - Immediate medical consultation necessary!")
            print("   • Seek immediate medical attention")
            print("   • Consider emergency evaluation")
            print("   • Implement all preventive measures")
        
        print(f"\n🏥 GENERAL HEALTH ADVICE:")
        print("   • Maintain a balanced, low-sodium diet")
        print("   • Engage in regular physical activity (30 min/day)")
        print("   • Avoid tobacco and limit alcohol consumption")
        print("   • Manage stress through relaxation techniques")
        print("   • Monitor and control blood pressure")
        print("   • Keep blood glucose levels in normal range")
        print("   • Maintain healthy weight (BMI 18.5-24.9)")
    
    def batch_prediction_demo(self):
        """Demonstrate batch prediction with sample data"""
        print("\n" + "="*60)
        print("📋 BATCH PREDICTION DEMONSTRATION")
        print("="*60)
        
        # Sample patients for demonstration
        sample_patients = [
            {
                'gender': 'Male',
                'age': 65,
                'hypertension': 1,
                'heart_disease': 1,
                'ever_married': 'Yes',
                'work_type': 'private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 180.0,
                'bmi': 32.5,
                'smoking_status': 'formerly smoked'
            },
            {
                'gender': 'Female',
                'age': 35,
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'No',
                'work_type': 'private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 95.0,
                'bmi': 22.0,
                'smoking_status': 'never smoked'
            },
            {
                'gender': 'Male',
                'age': 55,
                'hypertension': 1,
                'heart_disease': 0,
                'ever_married': 'Yes',
                'work_type': 'self-employed',
                'Residence_type': 'Rural',
                'avg_glucose_level': 140.0,
                'bmi': 28.0,
                'smoking_status': 'smokes'
            }
        ]
        
        print(f"Processing {len(sample_patients)} sample patients...\n")
        
        for i, patient in enumerate(sample_patients, 1):
            print(f"👤 PATIENT {i}:")
            print(f"   Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"   Hypertension: {patient['hypertension']}, Heart Disease: {patient['heart_disease']}")
            print(f"   BMI: {patient['bmi']}, Glucose: {patient['avg_glucose_level']} mg/dL")
            
            result = self.predict_single_patient(patient)
            if result:
                risk_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}
                print(f"   Result: {risk_emoji.get(result['risk_level'], '❓')} {result['risk_level']} Risk")
                print(f"   Stroke Probability: {result['probability_stroke']:.1%}")
            else:
                print("   ❌ Prediction failed")
            print()

def main():
    """Main application function"""
    app = StrokePredictionApp()
    
    print("🏥 STROKE PREDICTION SYSTEM")
    print("=" * 50)
    print("K-Nearest Neighbors Model - Medical Grade Predictions")
    print("=" * 50)
    
    # Try to load the trained model
    if not app.load_trained_model():
        print("\n❌ Cannot start application without trained model.")
        print("Please run the Feature.ipynb notebook first to train the KNN model.")
        return
    
    while True:
        print(f"\n📋 MAIN MENU")
        print("-" * 20)
        print("1. 🔍 Interactive Prediction")
        print("2. ℹ️  Model Information")
        print("3. 🚪 Exit")
        
        try:
            choice = input("\nChoose an option (1-3): ").strip()
            
            if choice == '1':
                app.interactive_prediction()
                
            elif choice == '2':
                print(f"\n📊 MODEL INFORMATION")
                print("-" * 25)
                print(f"Algorithm: K-Nearest Neighbors")
                print(f"Features: {len(app.selected_features) if app.selected_features else 'N/A'}")
                print(f"Model Type: Classification")
                print(f"Optimization: Medical-focused (High Recall)")
                print(f"Performance: Optimized for stroke risk assessment")
                print(f"Use Case: Medical prediction system")
                
            elif choice == '3':
                print("\n👋 Thank you for using the Stroke Prediction System!")
                print("Remember: This tool is for educational purposes only.")
                print("Always consult healthcare professionals for medical decisions.")
                break
                
            else:
                print("❌ Invalid option. Please choose between 1 and 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()