import pandas as pd

ADMISSION = "Admission"
DISCHARGE = "Discharge"


def transform_patient_info(
    df: pd.DataFrame,
    pid: str = "subject_id",
    dob: str = "DOB",
    dod: str = "DOD",
    time="time",
    code="code",
) -> pd.DataFrame:
    """
    Transform the standard patient info format, into eventstream (MEDS) format.
    Assumes that the DataFrame has columns: PID, DOB, DOD, as well as other columns describing static features.
    """
    required_columns = [pid, dob, dod]
    check_required_columns(df, required_columns)

    date_melted = df.melt(
        id_vars=[pid], value_vars=[dob, dod], var_name=code, value_name=time
    )

    # Drop rows where TIMESTAMP is NaT/None (for DEATHDATE)
    date_melted = date_melted.dropna(subset=[time])

    # Create a new DataFrame for the non-date columns
    non_date_df = df[[pid]].copy()

    # Combine each non-date column name and value into a new column
    for column in df.columns.difference(required_columns):
        non_date_df[column] = df[column].apply(
            lambda x: f"{column.upper()}//{x.upper()}"
        )

    # Melt the DataFrame to long format
    non_date_melted = non_date_df.melt(id_vars=[pid], value_name=code).drop(
        columns="variable"
    )

    # Combine the date-related and non-date-related DataFrames
    final_df = pd.concat([date_melted, non_date_melted], ignore_index=True)
    final_df.sort_values(by=[pid, time], inplace=True, na_position="first")
    return final_df


def transform_admission_discharge(
    df: pd.DataFrame,
    pid: str = "subject_id",
    admission: str = "ADMISSION_ID",
    time: str = "time",
    code: str = "code",
) -> pd.DataFrame:
    """
    Transforms the admission_id column into 'Admission' and 'Discharge' events.
    'Admission' is created at the first occurrence of an admission_id for a subject_id,
    and 'Discharge' is created at the last occurrence of that admission_id.

    Assumes the DataFrame has the following columns: subject_id, time, admission_id, and code.
    """
    check_required_columns(df, [pid, admission, time, code])

    # Sort the dataframe by subject_id and time
    df = df.sort_values(by=[pid, time])

    # Create DataFrame for Admission (first occurrence of admission_id for each subject)
    first_occurrence = df.groupby([pid, admission], as_index=False).first()
    first_occurrence[code] = ADMISSION

    # Create DataFrame for Discharge (last occurrence of admission_id for each subject)
    last_occurrence = df.groupby([pid, admission], as_index=False).last()
    last_occurrence[code] = DISCHARGE

    # Combine the original DataFrame with the new Admission/Discharge rows
    result = pd.concat([df, first_occurrence, last_occurrence], ignore_index=True)
    result = result.drop(columns=[admission])
    result = result.drop_duplicates()
    result = sort_admission_and_discharge(result, pid, time, code)
    return result


def sort_admission_and_discharge(
    df: pd.DataFrame, pid: str = "subject_id", time: str = "time", code: str = "code"
) -> pd.DataFrame:
    """Sort the DataFrame by subject_id and time, with 'Admission' events first and 'Discharge' events last with codes in between."""
    priority_mapping = {
        ADMISSION: 0,
        DISCHARGE: 2,
    }  # Admission should come first, Discharge last
    df["sort_priority"] = (
        df[code].map(priority_mapping).fillna(1)
    )  # Default priority for other events is 1
    # Sort by subject_id and time again to ensure the correct order
    df = df.sort_values(by=[pid, time, "sort_priority"], na_position='first').drop(columns="sort_priority")
    return df


def check_required_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Check if the DataFrame has the required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame must have the following columns: {', '.join(missing_columns)}"
        )


def sort_final_dataframe(
    df: pd.DataFrame,
    pid: str = "subject_id",
    time: str = "time",
    code: str = "code",
    dob: str = "DOB",
    dod: str = "DOD",
    other_static_vars: list = None,
) -> pd.DataFrame:
    """Sort the final DataFrame by subject_id and time, with 'Admission' events first and 'Discharge' events last with codes in between."""
    priority_mapping, code_priority = create_priority_mapping_dict(dob, dod, other_static_vars)
    df["sort_priority"] = (
        df[code].map(priority_mapping).fillna(code_priority)
    )  # Default priority for other events is 1
    # Sort by subject_id and time again to ensure the correct order
    df = df.sort_values(by=[pid, time, "sort_priority"], na_position='first').drop(columns="sort_priority")
    return df

def create_priority_mapping_dict(dob: str = "DOB", dod: str = "DOD", other_static_vars: list = None):
    """
    Create a priority mapping dictionary for sorting the DataFrame.
    Returns priorities for each static var and the codes 'Admission' and 'Discharge'.
    Other codes get a default priority value.
    """
    if other_static_vars is not None:
        other_static_vars = []
    priority_mapping = {
        dob: 0,
    }
    for var in other_static_vars:
        priority_mapping[var] = len(priority_mapping)
    priority_mapping[ADMISSION] = len(priority_mapping)
    code_priority = len(priority_mapping)
    priority_mapping[DISCHARGE] = len(priority_mapping) + 1
    priority_mapping[dod] = len(priority_mapping) + 1
    return priority_mapping, code_priority

