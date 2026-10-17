"""
Configuration settings for the NHANES Causal Time Series Analysis Project.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# CDC NHANES Base URL
CDC_NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

# Survey Cycles and their file suffixes
SURVEY_CYCLES = {
    "1999-2000": {"suffix": "", "year": 2000},
    "2001-2002": {"suffix": "_B", "year": 2002},
    "2003-2004": {"suffix": "_C", "year": 2004},
    "2005-2006": {"suffix": "_D", "year": 2006},
    "2007-2008": {"suffix": "_E", "year": 2008},
    "2009-2010": {"suffix": "_F", "year": 2010},
    "2011-2012": {"suffix": "_G", "year": 2012},
    "2013-2014": {"suffix": "_H", "year": 2014},
    "2015-2016": {"suffix": "_I", "year": 2016},
    "2017-2018": {"suffix": "_J", "year": 2018},
}

# Standard Core Variables
VARIABLE_MAPPINGS = {
    "SEQN": "respondent_id",
    "RIDAGEYR": "age",
    "RIAGENDR": "gender",  # 1 = Male, 2 = Female
    "RIDRETH1": "race_ethnicity",
    "WTINT2YR": "interview_weight",
    "WTMEC2YR": "exam_weight",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_circumference",
    "BPXSY1": "systolic_bp",
    "BPXDI1": "diastolic_bp",
    "LBXGH": "hba1c",
    "LBXTC": "total_cholesterol",
    "LBDHDD": "hdl_cholesterol",
    "DR1TKCAL": "daily_calories",
    "DR1TSUGR": "daily_sugar_g",
    "PAD680": "sedentary_minutes",
}

# Analysis Settings
RANDOM_SEED = 42
SIGNIFICANCE_ALPHA = 0.05
MAX_VAR_LAGS = 2
FORECAST_HORIZON = 8
