""" script to preprocess the data. WIll save the preprocessed csv in current directory 

usage example :

python3 preprocess.py /path/to/train.csv /path/to/test.csv --subsample_head 100000

By default, only uses the first 10k lines of the csvs for a quick demo. Can be changed with --subsample_head
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import argparse


import warnings
warnings.filterwarnings("ignore")

# preprocessing functions :

def drop_columns(df):
    """Drop les colonnes définies ci dessous
    
    Retourne le df modifié
    """    
    
    columns_to_drop = ['piezo_station_commune_code_insee', 'piezo_station_department_code', 'meteo_name', 
                        'piezo_station_bss_id', 'prelev_structure_code_0', 'prelev_structure_code_1', 
                        'prelev_structure_code_2', 'piezo_station_pe_label', 'hydro_method_label','piezo_status',
                         'piezo_station_bdlisa_codes','hydro_station_code','piezo_station_department_name', 
                         'piezo_station_commune_name', 'piezo_measure_nature_name', 'meteo_name', 'hydro_status_label', 
                         'piezo_station_bss_code', 'piezo_bss_code', 'piezo_producer_name', 'meteo_id','hydro_method_code','piezo_continuity_code',
                         'hydro_hydro_quantity_elab']

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def CleanNA_preprocessing(dataset_input,CategoricalAddNA_threshold=15, numericalCleanNA_threshhold=70, numerical_mode="median"):
    """
    input :
        - ategoricalAddNA_threshold = 15
        - numericalCleanNA_threshhold = 70
        - numerical_mode="median"
    """

    dataset_output = dataset_input

    ## convert 4 insee collumns to float :
    dataset_output["insee_%_const"] = pd.to_numeric(dataset_output["insee_%_const"], errors='coerce')
    dataset_output["insee_%_ind"] = pd.to_numeric(dataset_output["insee_%_ind"], errors='coerce')
    dataset_output["insee_med_living_level"] = pd.to_numeric(dataset_output["insee_med_living_level"], errors='coerce')
    dataset_output["insee_%_agri"] = pd.to_numeric(dataset_output["insee_%_agri"], errors='coerce')

    ## Variables categorical :
    # Liste des colonnes avec des valeurs manquantes à traiter
    columns_to_impute = [
        'prelev_volume_obtention_mode_label_2', 'prelev_usage_label_2',
        'prelev_usage_label_1', 'prelev_volume_obtention_mode_label_1',
        'prelev_usage_label_0', 'prelev_volume_obtention_mode_label_0',
        'piezo_measure_nature_code'
    ]
    for col in columns_to_impute:
        # Calculate the percentage of missing values in the column
        percentage_na = dataset_output[col].isna().mean() * 100
        
        if percentage_na > CategoricalAddNA_threshold:
            # Replace all missing values with "na" if percentage > 15%
            dataset_output[col].fillna("na", inplace=True)
        else:
            # Otherwise, replace missing values with the mode
            mode = dataset_output[col].mode()[0]  # Calculate the mode (most frequent value)
            dataset_output[col].fillna(mode, inplace=True)


    ## Variables numerical :
    # Filter only numerical columns
    numerical_data = dataset_output.select_dtypes(include=['float64', 'int64'])
    # Calculate the percentage of missing values for each column
    missing_percentage = (numerical_data.isnull().sum() / len(dataset_output)) * 100
    # Create a list of numerical columns with missing values
    columns_with_missing_values = missing_percentage[missing_percentage > 0].index.tolist()
    # Handle missing values: Drop columns with >=70% missing, otherwise fill with the median
    for col in columns_with_missing_values:
        if missing_percentage[col] >= numericalCleanNA_threshhold:  # Correctly reference the missing percentage of the specific column
            dataset_output.drop(columns=[col], inplace=True)
        else:
            if numerical_mode == "mean":
                dataset_output[col].fillna(dataset_output[col].mean(), inplace=True)
            else:
                dataset_output[col].fillna(dataset_output[col].median(), inplace=True)


    return dataset_output


def add_distance_from_algiers(dataset):

    # Define Algiers coordinates (latitude and longitude in degrees)
    algiers_latitude = 47.0812  # Latitude of Algiers
    algiers_longitude = 2.3980  # Longitude of Algiers

    # Convert degrees to radians
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth using the Haversine formula.
        """
        # Earth radius in kilometers
        R = 6371.0  

        # Convert latitude and longitude to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Distance in kilometers
        distance = R * c
        return distance

    # Apply the Haversine formula to calculate distance from Algiers for each row
    dataset['distance_from_algiers'] = dataset.apply(
        lambda row: haversine(
            row['piezo_station_latitude'], row['piezo_station_altitude'],
            algiers_latitude, algiers_longitude
        ),
        axis=1
    )


    return dataset

def DateTime_Preprocessing_importupdatefromCSV(dataset_input,piezo_station_update_date__csv_path = "df_piezo_station_update_date.csv"): #, piezo_station_update_date__csv_path="df_piezo_station_update_date.csv"):
    """
        input :
            - piezo_station_update_date__csv_path = "df_piezo_station_update_date.csv" : path to csv of df_piezo_station_update_date collumn that massyl have
        output :
            - collumn : "measurement_day_of_year"
            - collumn : "measurement_year"
    """

    dataset_output = dataset_input


    ## drop "hydro_observation_date_elab" and "meteo_date"
    dataset_output.drop(columns=["hydro_observation_date_elab","meteo_date","piezo_station_update_date"],inplace=True)


    ## to_datetime :
    dataset_output['piezo_measurement_date'] = pd.to_datetime(dataset_output['piezo_measurement_date'])


    ## Preprocess "piezo_measurement_date" :
    # Calculate the fraction of the year for mm-dd
    dataset_output['measurement_day_of_year'] = dataset_output['piezo_measurement_date'].dt.day_of_year  # Day number within the year (1-366)
    dataset_output['measurement_day_of_year'] = dataset_output['measurement_day_of_year'] / 366  # Normalize to [0, 1]
    # Extract the year
    dataset_output['measurement_year'] = dataset_output['piezo_measurement_date'].dt.year
    # Encode "measurement_year"
    mapping = {
        2020: 0,
        2021: 1,
        2022: 2,
        2023: 3
    }
    dataset_output['measurement_year'] = dataset_output['measurement_year'].replace(mapping)
    # drop the transformed "piezo_measurement_date"
    dataset_output.drop(columns=["piezo_measurement_date"],inplace=True)

    return dataset_output

def encode(df, encoding='label'):
    """ encode les variables évidemment ordonnées en ordinal, les variables binaires et
    (au choix) encode les variables non ordonnées soit via OHE soit via label encoding
    
    Variables
    ---------
    encoding (str) : 'label' pour label encoding, 'OHE' pour one hot (uniquement variables pertinentes)
    """

    mappings_ord = {
        'hydro_qualification_label' : {'Douteuse': 0, 'Non qualifiée': 1, 
                           'Bonne': 2},
        'piezo_qualification' : {'Incorrecte': 0, 'Incertaine': 1, 
                           'Non qualifié': 2, 'Correcte': 3},
        'piezo_obtention_mode' : {'Valeur mesurée':1, "Mode d'obtention inconnu":0,
                                'Valeur reconstituée':-1}
        
        }
    #binaire, à label encoder
    columns_to_encode_binaire = ['hydro_hydro_quantity_elab',
                                'piezo_continuity_name',]

    # soit label-encode soit one hot-encode
    columns_to_encode = ['piezo_measure_nature_code',
                        'prelev_usage_label_0', 
                        'prelev_usage_label_1', 
                        'prelev_usage_label_2',
                        'prelev_volume_obtention_mode_label_0', 
                        'prelev_volume_obtention_mode_label_1', 
                        'prelev_volume_obtention_mode_label_2',] #target
    label_encoders = {}
    
    #ordinal, on ordonne
    for col, mapping in mappings_ord.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # encoding au choix
    if encoding == 'OHE':
        columns_to_LE = columns_to_encode_binaire
        #OHE
        for col in columns_to_encode:
            if col in df.columns:
                one_hot = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df, one_hot], axis=1)
                df = df.drop(columns=[col], errors='ignore') 
    else:
        columns_to_LE = columns_to_encode_binaire + columns_to_encode
        
    #LE
    for col in columns_to_LE:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le 
    
    return df

# encoding labels
def encode_y(df):
    df['piezo_groundwater_level_category'] = df['piezo_groundwater_level_category'].map({'Very Low': 0, 'Low': 1, 'Average': 2, 'High': 3, 'Very High': 4})
    return df
    
def standardise_preprocessing(dataset_input):
    # Create a StandardScaler instance
    scalers = {}

    for column in dataset_input.columns:
        scaler = StandardScaler()
        # Standardize only numerical columns with more than 10 unique values
        if dataset_input[column].dtype in ['int64', 'float64'] and dataset_input[column].nunique() > 10:
            # Apply StandardScaler
            dataset_input[column] = scaler.fit_transform(dataset_input[[column]])
            scalers[column] = scaler

    return dataset_input, scalers

def standardise_preprocessing_test(dataset_input, scalers):
    # Create a StandardScaler instance

    for column in scalers.keys():
        scaler = scalers[column]
        # Standardize only numerical columns with more than 10 unique values
        dataset_input[column] = scaler.transform(dataset_input[[column]])

    return dataset_input

def remove_colinear_collumns_preprocessing(dataset_input, correlation_threshold = 0.8) :
    # Calculate the correlation matrix
    correlation_matrix = dataset_input.corr()

    # Mask the upper triangle and the diagonal (convert mask to boolean)
    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)

    # Apply the mask
    filtered_corr = correlation_matrix.where(mask)

    # Identify columns to drop
    columns_to_drop = set()
    for col1 in filtered_corr.columns:
        for col2 in filtered_corr.columns:
            if not np.isnan(filtered_corr.loc[col1, col2]) and abs(filtered_corr.loc[col1, col2]) > correlation_threshold:
                # Add one of the columns to the drop list
                columns_to_drop.add(col2)

    # Drop the identified columns
    dataset_input = dataset_input.drop(columns=columns_to_drop)

    return dataset_input


def remove_low_variance_columns(dataset_input, variance_threshold=0.01):
    # Calculate the variance for each column
    variances = dataset_input.var()

    # Identify columns to drop
    columns_to_drop = [column for column in variances.index if variances[column] < variance_threshold]

    # Drop the identified columns
    dataset_input = dataset_input.drop(columns=columns_to_drop)

    # print
    print("columns_to_drop : ",columns_to_drop)

    return dataset_input

# decode labels
def decode_y(df):
    forward = {'Very Low': 0, 'Low': 1, 'Average': 2, 'High': 3, 'Very High': 4}
    backward = {v: k for k, v in forward.items()}
    df['piezo_groundwater_level_category'] = df['piezo_groundwater_level_category'].map(backward)
    return df

def upsample_summer_preprocessing(df):
    """ prend un dataframe et dedouble les valeurs d'ete"""
    #détecter ete

    debut = (28 * 5) / 366  # on laisse un peu de jours fin mai
    fin = (365 - (28 * 3)) / 366 # et début octobre

    #index of the summer data
    ix_summer = (df['measurement_day_of_year'] >= debut) & (df['measurement_day_of_year'] <= fin)

    rows_to_duplicate = df.loc[ix_summer]
    new_indexes = range(9999999, 9999999 - len(rows_to_duplicate), -1)
    duplicated_rows = rows_to_duplicate.copy()
    duplicated_rows.index = new_indexes
    duplicated_rows.index.name = "row_index"
    df = pd.concat([df, duplicated_rows])

    return df, ix_summer

def upsample_summer_preprocessing_y(df, ix_summer):

    rows_to_duplicate = df.loc[ix_summer]
    new_indexes = range(9999999, 9999999 - len(rows_to_duplicate), -1)
    duplicated_rows = rows_to_duplicate.copy()
    duplicated_rows.index = new_indexes
    duplicated_rows.index.name = "row_index"
    df = pd.concat([df, duplicated_rows])

    return df

def upsample_plus2_summer_preprocessing(df):
    """Takes a dataframe and triples the summer values."""
    # Define summer period
    debut = (28 * 5) / 366  # End of May
    fin = (365 - (28 * 3)) / 366  # Beginning of October

    # Index of the summer data
    ix_summer = (df['measurement_day_of_year'] >= debut) & (df['measurement_day_of_year'] <= fin)

    # Select rows to duplicate
    rows_to_duplicate = df.loc[ix_summer]
    
    # Create two sets of duplicates with new indexes
    duplicated_rows_1 = rows_to_duplicate.copy()
    duplicated_rows_1.index = range(9999999, 9999999 - len(rows_to_duplicate), -1)
    
    duplicated_rows_2 = rows_to_duplicate.copy()
    duplicated_rows_2.index = range(9999999 - len(rows_to_duplicate), 9999999 - (2 * len(rows_to_duplicate)), -1)
    
    # Concatenate the original dataframe with the two sets of duplicates
    df = pd.concat([df, duplicated_rows_1, duplicated_rows_2])

    return df, ix_summer


def upsample_plus2_summer_preprocessing_y(df, ix_summer):
    """Upsample the target variable by a factor of 3."""
    # Select rows to duplicate
    rows_to_duplicate = df.loc[ix_summer]
    
    # Create two sets of duplicates with new indexes
    duplicated_rows_1 = rows_to_duplicate.copy()
    duplicated_rows_1.index = range(9999999, 9999999 - len(rows_to_duplicate), -1)
    
    duplicated_rows_2 = rows_to_duplicate.copy()
    duplicated_rows_2.index = range(9999999 - len(rows_to_duplicate), 9999999 - (2 * len(rows_to_duplicate)), -1)
    
    # Concatenate the original dataframe with the two sets of duplicates
    df = pd.concat([df, duplicated_rows_1, duplicated_rows_2])

    return df

def filter_out(df):
    """ete dans x_test : du 01/06 au 30/09"""
    debut = (28 * 5) / 366  # on laisse un peu de jours fin mai
    fin = (365 - (28 * 3)) / 366 # et début octobre

    df = df[(df['measurement_day_of_year'] >= debut) & (df['measurement_day_of_year'] <= fin)]

    return df

def PreProcessing_full(dataset_X_train, dataset_X_test, CategoricalAddNA_threshold=15, numericalCleanNA_threshhold=70, filter_sum=False, encoding='label',correlation_threshold = 0.8, upsample_summer=0):
    # Optional filtering for non-summer measurements
    if filter_sum:
        dataset_X_train = filter_out(dataset_X_train)  # Implement filter_out function
        dataset_X_test = filter_out(dataset_X_test)

    # Separate y_train from X_train
    X_train = dataset_X_train.drop(columns=['piezo_groundwater_level_category'])  # Features
    y_train = pd.DataFrame(dataset_X_train['piezo_groundwater_level_category'])  # Target

    # Encode y_train
    y_train_preprocessed = encode_y(y_train)  # Implement encode_y function

    # save index
    X_train_index = dataset_X_train.index
    X_test_index = dataset_X_test.index

    # Concatenate X_train and X_test
    dataset_train_test = pd.concat([X_train, dataset_X_test], axis=0, ignore_index=True)

    # Preprocess the dataset
    dataset_train_test = drop_columns(dataset_train_test)  # Implement drop_columns function
    dataset_train_test = CleanNA_preprocessing(dataset_train_test, CategoricalAddNA_threshold, numericalCleanNA_threshhold)  # Implement CleanNA_preprocessing
    dataset_train_test = add_distance_from_algiers(dataset_train_test)
    dataset_train_test = DateTime_Preprocessing_importupdatefromCSV(dataset_train_test)  # Implement DateTime_Preprocessing
    dataset_train_test = encode(dataset_train_test, encoding)  # Implement encode function
    dataset_train_test, scalers = standardise_preprocessing(dataset_train_test)  # Implement standardise_preprocessing
    dataset_train_test = remove_colinear_collumns_preprocessing(dataset_train_test, correlation_threshold)

    # Split preprocessed dataset back into train and test sets
    X_train_preprocessed = dataset_train_test.iloc[:len(X_train), :]  # Train part
    X_test_preprocessed = dataset_train_test.iloc[len(X_train):, :]  # Test part

    # Restore original index
    X_train_preprocessed.index = X_train_index
    X_test_preprocessed.index = X_test_index

    # upsampling summer :
    if upsample_summer == 1 :
        X_train_preprocessed, ix_summer = upsample_summer_preprocessing(X_train_preprocessed)
        y_train_preprocessed = upsample_summer_preprocessing_y(y_train_preprocessed, ix_summer)
    elif upsample_summer == 2 :
        X_train_preprocessed, ix_summer = upsample_plus2_summer_preprocessing(X_train_preprocessed)
        y_train_preprocessed = upsample_plus2_summer_preprocessing_y(y_train_preprocessed, ix_summer)

    # Save preprocessed data to CSV
    X_train_preprocessed.to_csv('X_train_preprocessed.csv', sep=',', index=True)
    y_train_preprocessed.to_csv('y_train_preprocessed.csv', sep=',', index=True)
    X_test_preprocessed.to_csv('X_test_preprocessed.csv', sep=',', index=True)

    return X_train_preprocessed, y_train_preprocessed, X_test_preprocessed

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Parses the data to preprocess')
    
    # Add arguments
    parser.add_argument('path_train', type=str, help='Path to train csv')
    parser.add_argument('path_test', type=str, help='Path to test csv')
    parser.add_argument('--subsample_head', type=int, default=10000, 
                        help='Number of rows to subsample from the head (default: 10000)')
    
    # Parse the arguments
    args = parser.parse_args()

    #read
    dataset_X_train = pd.read_csv(args.path_train, index_col="row_index", nrows=args.subsample_head)
    dataset_X_test = pd.read_csv(args.path_test,index_col="row_index", nrows=args.subsample_head)

    #preprocess
    X_train_preprocessed, y_train_preprocessed, X_test_preprocessed = PreProcessing_full(dataset_X_train, dataset_X_test, CategoricalAddNA_threshold=15, numericalCleanNA_threshhold=70, filter_sum=False, encoding='label')

if __name__ == '__main__':
    main()

    