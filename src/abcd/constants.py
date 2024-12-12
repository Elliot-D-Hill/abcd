import polars as pl

EVENTS = [
    "baseline_year_1_arm_1",
    "1_year_follow_up_y_arm_1",
    "2_year_follow_up_y_arm_1",
    "3_year_follow_up_y_arm_1",
    "4_year_follow_up_y_arm_1",
]
EVENTS_TO_VALUES = dict(zip(EVENTS, range(len(EVENTS))))
RACE_MAPPING = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
SEX_MAPPING = {0: "Female", 1: "Male"}
COLUMNS = {
    "split": "Split",
    "src_subject_id": "Subject ID",
    "eventname": "Follow-up event",
    "y_t": "Quartile at t",
    "y_{t+1}": "Quartile at t+1",
    "demo_sex_v2": "Sex",
    "race_ethnicity": "Race",
    "interview_age": "Age",
    "interview_date": "Event year",
    "adi_percentile": "ADI quartile",
    "parent_highest_education": "Parent highest education",
    "demo_comb_income_v2": "Combined income",
}
EVENT_NAMES = ["Baseline", "1-year", "2-year", "3-year"]
EVENTS_TO_NAMES = dict(zip(EVENTS, EVENT_NAMES))
GROUP_ORDER = pl.Enum(
    [
        "Conversion",
        "Persistence",
        "Agnostic",
        "1",
        "2",
        "3",
        "4",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "Baseline",
        "1-year",
        "2-year",
        "3-year",
        "Asian",
        "Black",
        "Hispanic",
        "White",
        "Other",
        "Female",
        "Male",
    ]
)
RISK_GROUPS = {
    "1": "No risk",
    "2": "Low risk",
    "3": "Moderate risk",
    "4": "High risk",
}
