import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')
X_cols = ['fare', 'in-vehicle time', 'waiting time', 'access & egress time',
          'transfer time', 'crowding level', 'customs clearance time']

distances = ['Short (50km)', 'Medium (100km)', 'Long (150km)']
time_vars = ['in-vehicle time', 'access & egress time', 'customs clearance time']
# =====================================================================
# 1. DATA PREPARATION & CLEANING
# =====================================================================
def load_and_clean_data():
    print("Loading survey data...")
    survey = pd.read_excel('GBA Final Data (1).xlsx', sheet_name='Full')
    survey_data = survey.drop(index=0).reset_index(drop=True)
    survey_data['QNSet'] = pd.to_numeric(survey_data['QNSet'], errors='coerce')

    # Identify the Income column (Column name '5')
    income_col = [c for c in survey_data.columns if str(c) == '5']
    if income_col:
        survey_data['Income'] = pd.to_numeric(survey_data[income_col[0]], errors='coerce')
    else:
        print("Warning: Could not find Income column '5'.")

   

    def clean_attributes(df):
        df = df.copy()
        df['scenario_id'] = df['scenario_id'].apply(lambda x: f"{float(x):.1f}")
        df = df.replace('--', '0')
        if 'crowding level' in df.columns:
            df['crowding level'] = df['crowding level'].astype(str).str.replace('%', '')
            df['crowding level'] = pd.to_numeric(df['crowding level'], errors='coerce').fillna(0) / 100.0
        for col in X_cols:
            if col in df.columns and col != 'crowding level':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    attr_files = {
        1: 'data_mode_attributes(0).csv',
        2: 'data_mode_attributes(1).csv',
        3: 'data_mode_attributes(2).csv',
        4: 'data_mode_attributes(3).csv'
    }

    attr_dicts = {}
    for qnset, fname in attr_files.items():
        if os.path.exists(fname):
            attr_dicts[qnset] = clean_attributes(pd.read_csv(fname))

    return survey_data, attr_dicts, X_cols

# =====================================================================
# 2. DATA POOLING (WITH ASC DUMMY VARIABLES)
# =====================================================================
def create_long_format(survey_data, attr_dicts):
    print("Pooling data and generating Alternative Specific Constants (ASCs)...")
    long_records = []

    for qn in [1, 2, 3, 4]:
        if qn not in attr_dicts: continue
        subset = survey_data[survey_data['QNSet'] == qn]
        attr_df = attr_dicts[qn]

        for s in range(1, 13):
            for v in [1, 2]:
                col_name = f"{s}.{v}"
                if col_name in subset.columns:
                    choices = pd.to_numeric(subset[col_name], errors='coerce').dropna().astype(int)
                    sc_attrs = attr_df[attr_df['scenario_id'] == col_name]

                    for choice in choices:
                        for mode_idx in range(1, 6):
                            mode_row = sc_attrs[sc_attrs['mode'] == mode_idx]
                            if not mode_row.empty:
                                record = mode_row.iloc[0].to_dict()
                                record['is_chosen'] = 1 if mode_idx == choice else 0
                                record['Purpose'] = 'Work' if v == 1 else 'Non-Work'
                                
                                # Generate ASCs (Mode 1: Bus/MTR is the baseline/reference)
                                record['ASC_HSR'] = 1 if mode_idx == 2 else 0
                                record['ASC_Taxi'] = 1 if mode_idx == 3 else 0
                                record['ASC_PrivateCar'] = 1 if mode_idx == 4 else 0
                                record['ASC_eVTOL'] = 1 if mode_idx == 5 else 0

                                long_records.append(record)

    df_long = pd.DataFrame(long_records)

    # Categorize distances
    def categorize_distance(scenario_id):
        sc_num = float(str(scenario_id).split('.')[0])
        if 1 <= sc_num <= 4: return 'Short (50km)'
        elif 5 <= sc_num <= 8: return 'Medium (100km)'
        else: return 'Long (150km)'

    df_long['Distance_Category'] = df_long['scenario_id'].apply(categorize_distance)
    return df_long

# =====================================================================
# 3. CORE REGRESSION ENGINE (UNSCALING COEFFICIENTS)
# =====================================================================
def run_regression(df_subset, features):
    X = df_subset[features]
    y = df_subset['is_chosen']

    # Scale features for L1 solver stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run Model
    model = LogisticRegression(penalty='l1', solver='saga', C=5.0, max_iter=5000, random_state=42)
    model.fit(X_scaled, y)

    # UNSCALE coefficients to get true economic values
    raw_coefs = model.coef_[0] / scaler.scale_
    return dict(zip(features, raw_coefs))

# =====================================================================
# 4. MAIN EXECUTION (ANALYSIS 1 & 2)
# =====================================================================
def calculate_vtts_table(df):
    vtts_results = {}
    for dist in distances:
        subset = df[df['Distance_Category'] == dist]
        coefs = run_regression(subset, X_cols)
        beta_fare = coefs['fare']

        res = {}
        for t_col in time_vars:
            if beta_fare < 0:
                res[t_col] = (coefs[t_col] / beta_fare) * 60
            else:
                res[t_col] = np.nan
        vtts_results[dist] = res
    return pd.DataFrame(vtts_results).round(2)
def main():
    survey_data, attr_dicts, X_cols = load_and_clean_data()
    df_long_all = create_long_format(survey_data, attr_dicts)

    if df_long_all.empty:
        print("Error: No data pooled.")
        return
    df_long_work = df_long_all[df_long_all['Purpose'] == 'Work'].copy()
    df_long_non_work = df_long_all[df_long_all['Purpose'] == 'Non-Work'].copy()
    # Create High Income Subset (Income options 3, 4, 5 represent >50k HKD)
    high_income_survey = survey_data[survey_data['Income'] >= 3].copy()
    df_long_high = create_long_format(high_income_survey, attr_dicts)

    print("\n" + "="*50)
    print("ANALYSIS 1: VALUE OF TRAVEL TIME SAVINGS (VTTS)")
    print("="*50)


    def calculate_vtts_table(df):
        vtts_results = {}
        for dist in distances:
            subset = df[df['Distance_Category'] == dist]
            coefs = run_regression(subset, X_cols)
            beta_fare = coefs['fare']

            res = {}
            for t_col in time_vars:
                # VTTS = (Beta_Time / Beta_Fare) * 60. Only valid if Beta_Fare is negative.
                if beta_fare < 0:
                    res[t_col] = (coefs[t_col] / beta_fare) * 60
                else:
                    res[t_col] = np.nan
            vtts_results[dist] = res
        return pd.DataFrame(vtts_results).round(2)

    print("\n--- VTTS: General Population (HKD/Hour) ---")
    print(calculate_vtts_table(df_long_all).to_markdown())

    print("\n--- VTTS: High Income Bracket (>50k HKD) ---")
    print(calculate_vtts_table(df_long_high).to_markdown())

    print("\n--- VTTS: Work Trips (HKD/Hour) ---")
    print(calculate_vtts_table(df_long_work).to_markdown())
    print("\n--- VTTS: Non-Work Trips (HKD/Hour) ---")
    print(calculate_vtts_table(df_long_non_work).to_markdown())
    print("\n\n" + "="*50)
    print("ANALYSIS 2: ALTERNATIVE SPECIFIC CONSTANTS (ASCs)")
    print("="*50)
    print("Note: Mode 1 (Bus/MTR) is the baseline (ASC = 0).")
    print("Positive ASC = Inherent preference. Negative ASC = Inherent aversion/fear.\n")


    asc_features = X_cols + ['ASC_HSR', 'ASC_Taxi', 'ASC_PrivateCar', 'ASC_eVTOL']

    asc_results = {}
    for dist in distances:
        subset = df_long_all[df_long_all['Distance_Category'] == dist]
        coefs = run_regression(subset, asc_features)

        # Extract just the ASCs for the report
        asc_results[dist] = {
            'HSR (Mode 2)': coefs['ASC_HSR'],
            'Taxi (Mode 3)': coefs['ASC_Taxi'],
            'Private Car (Mode 4)': coefs['ASC_PrivateCar'],
            'eVTOL (Mode 5)': coefs['ASC_eVTOL']
        }

    asc_df = pd.DataFrame(asc_results).round(4)
    print(asc_df.to_markdown())

    # Export to CSV for your FYP report tables
    asc_df.to_csv("GBA_ASC_Results.csv")
    print("\n=> ASC results saved to 'GBA_ASC_Results.csv' for your report.")

if __name__ == "__main__":
    main()
