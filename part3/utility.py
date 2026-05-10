import os
import pandas as pd


def clean_exam_data(df):
    """Clean the exam data by converting percentage strings to floats and calculating a final score."""
    # Find columns that might have percentage strings (excluding StudentID)
    exam_cols = [col for col in df.columns if col != 'StudentID']
    
    for col in exam_cols:
        if df[col].dtype == object:
            # Remove '%' and convert to float
            df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100.0
            
    # Calculate Final_Exam_Score as the mean of the topic columns
    df['Final_Exam_Score'] = df[exam_cols].mean(axis=1)
    return df

def get_data(data_folder:str = 'data'):
    """Load and merge the data from the given folder.

    Args:
        data_folder (str): The folder where the data is located.

    Returns:
        pd.DataFrame: A merged dataframe containing all students' quiz and exam scores.
    """

    semesters = set()
    for filename in os.listdir(data_folder):
        if filename.startswith('Semester'):
            semester = filename.split('_')[0]
            semesters.add(semester)

    all_semesters_df = []

    for semester in semesters:
        quiz_file = os.path.join(data_folder, f'{semester}_quiz_scores_anon.csv')
        exam_file = os.path.join(data_folder, f'{semester}_exam_scores_anon.csv')
        
        quiz_df = None
        exam_df = None
        
        if os.path.exists(quiz_file):
            print(f"Loading and cleaning quiz data for {semester}...")
            quiz_df = pd.read_csv(quiz_file, sep=';')
            quiz_df = clean_quiz_data(quiz_df)
            
        if os.path.exists(exam_file):
            print(f"Loading and cleaning exam data for {semester}...")
            exam_df = pd.read_csv(exam_file, sep=';')
            exam_df = clean_exam_data(exam_df)
            
        if quiz_df is not None and exam_df is not None:
            # Right join on StudentID to keep all students who took the exam (our target variable)
            print(f"Merging quiz and exam data for {semester}...")
            merged_semester = pd.merge(quiz_df, exam_df, on='StudentID', how='right')
            merged_semester['Semester'] = semester
            all_semesters_df.append(merged_semester)
            
    if all_semesters_df:
        final_df = pd.concat(all_semesters_df, ignore_index=True)
        
        # Fill NaN with 0 for 'Attempts' and 'Duration_minutes' columns
        zero_fill_cols = [c for c in final_df.columns if 'Attempts' in c or 'Duration_minutes' in c]
        if zero_fill_cols:
            final_df[zero_fill_cols] = final_df[zero_fill_cols].fillna(0)
            
        # Ensure Target is at the end (optional, but good practice)
        cols = final_df.columns.tolist()
        if 'Final_Exam_Score' in cols:
            cols.remove('Final_Exam_Score')
            cols.append('Final_Exam_Score')
            final_df = final_df[cols]
        return final_df
    
    return pd.DataFrame()

def clean_quiz_data(df):
    df = df.drop(columns=['Status', 'StartedAt', 'FinishedAt'])

    quiz_names = df['Quiz'].unique()
    quiz_dfs = {}

    for quiz_name in quiz_names:
        quiz_dfs[quiz_name] = df[df['Quiz'] == quiz_name].copy()

    for quiz_name, quiz_df in quiz_dfs.items():
        quiz_dfs[quiz_name] = quiz_df.dropna(axis=1, how='all')

    import re

    def duration_to_minutes(duration_str):
        """
        Converts a duration string (e.g., '23 Stunden 24 Minuten') to total minutes.
        Handles Stunden, Minuten, and Sekunden.
        """
        total_minutes = 0
        if isinstance(duration_str, str):
            hours_match = re.search(r'(\d+)\s*Stunden', duration_str)
            minutes_match = re.search(r'(\d+)\s*Minuten', duration_str)
            seconds_match = re.search(r'(\d+)\s*Sekunden', duration_str)

            if hours_match:
                total_minutes += int(hours_match.group(1)) * 60
            if minutes_match:
                total_minutes += int(minutes_match.group(1))
            if seconds_match:
                total_minutes += int(seconds_match.group(1)) / 60
        return total_minutes

    for quiz_name, quiz_df_orig in quiz_dfs.items():
        # Create a working copy to avoid modifying the original dictionary entry mid-loop before final assignment
        quiz_df = quiz_df_orig.copy()

        # 1. Drop attempts where duration is less than a minute
        quiz_df.loc[:, 'Duration_minutes'] = quiz_df['Duration'].apply(duration_to_minutes)
        quiz_df = quiz_df[quiz_df['Duration_minutes'] >= 1]

        # 2. Drop attempts where Score/10.00 is 0
        quiz_df = quiz_df[quiz_df['Score/10.00'] != 0]

        # 3. Transform 'F x /y.yy' columns to relative scores (0 to 1)
        # Identify columns that match the 'F x /y.yy' pattern
        fx_columns = [col for col in quiz_df.columns if re.match(r'^F\s\d+\s/\d+\.\d+$', col)]

        for col in fx_columns:
            try:
                # Extract the max score (y.yy) from the column name
                max_score = float(col.split('/')[-1])
                if max_score > 0:
                    quiz_df.loc[:, col] = quiz_df[col].astype(float) / max_score
                    # Rename the column to a simpler format
                    new_col_name = col.split('/')[0].strip()
                    quiz_df = quiz_df.rename(columns={col: new_col_name})
                else:
                    # If max_score is 0, these columns are usually all NaN or 0, so convert to NaN or leave as is if already processed.
                    # For simplicity, if max_score is 0, we can fill with 0 or NaN as it implies no score possible or relevant.
                    quiz_df.loc[:, col] = 0.0  # Or pd.NA or leave as is if already 0/NaN
                    new_col_name = col.split('/')[0].strip()
                    quiz_df = quiz_df.rename(columns={col: new_col_name})
            except (ValueError, IndexError):
                # Handle cases where parsing max_score fails or column name is malformed
                print(f"Warning: Could not process column {col} in quiz {quiz_name}")

        quiz_dfs[quiz_name] = quiz_df.copy()

    processed_quiz_dfs = {}

    for quiz_name, quiz_df in quiz_dfs.items():
        # Identify 'F x' columns
        fx_columns = [col for col in quiz_df.columns if re.match(r'^F\s\d+$', col)]

        # Ensure 'Score/10.00' and 'F x' columns are numeric
        quiz_df.loc[:, 'Score/10.00'] = pd.to_numeric(quiz_df['Score/10.00'], errors='coerce')
        for col in fx_columns:
            quiz_df.loc[:, col] = pd.to_numeric(quiz_df[col], errors='coerce')

        # Define aggregation dictionary
        agg_dict = {
            'Score/10.00': ['count', 'mean', 'max', 'min'],
            'Duration_minutes': ['sum', 'mean']
        }
        for col in fx_columns:
            agg_dict[col] = ['mean', 'max', 'min']

        # Group by StudentID and apply aggregations
        processed_df = quiz_df.groupby('StudentID').agg(agg_dict)

        # Flatten MultiIndex columns and rename for clarity
        processed_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in processed_df.columns.values]
        processed_df = processed_df.rename(columns={'Score/10.00_count': 'Attempts'})
        processed_df = processed_df.reset_index()

        processed_quiz_dfs[quiz_name] = processed_df

    merged_df = None

    for quiz_name, processed_df in processed_quiz_dfs.items():
        # Make a copy to avoid modifying the original dictionary entries
        df_to_merge = processed_df.copy()

        # Sanitize quiz_name for column prefix
        sanitized_quiz_name = quiz_name.replace(' ', '_').replace('/', '_').replace(':', '_').lower()

        # Rename columns to include quiz_name as prefix, except for 'StudentID'
        new_columns = {col: f'{sanitized_quiz_name}_{col}' for col in df_to_merge.columns if col != 'StudentID'}
        df_to_merge = df_to_merge.rename(columns=new_columns)

        if merged_df is None:
            merged_df = df_to_merge
        else:
            # Use an outer join to include all students from all quizzes
            merged_df = pd.merge(merged_df, df_to_merge, on='StudentID', how='outer')

    return merged_df