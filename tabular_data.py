import pandas as pd

def remove_rows_with_missing_ratings(dataframe: pd.DataFrame) -> pd.DataFrame:
    ratings_columns = [column for column in dataframe.columns if 'rate' in column]
    return dataframe.dropna(subset = ratings_columns)

def combine_description_strings(dataframe: pd.DataFrame, column:str) -> pd.DataFrame:
    dataframe = dataframe.dropna(subset = [column], axis = 0)
    clean_descriptions = list()
    list_of_descriptions = dataframe[column].to_list()
    for description in list_of_descriptions:
        description_list = string_to_list(description)
        description_string = ''.join(map(str, description_list))
        description_string = description_string.replace("'About this space',", ''
        ).replace('  ', ' '
        ).replace('"', ''
        ).replace("'", ''
        ).lstrip(' ')
        clean_descriptions.append(description_string)
    dataframe.loc[:, column] = clean_descriptions
    return dataframe

def string_to_list(string) -> list:
    def inner(item):
        while True:
            next_item = next(item)
            if next_item == '[':
                yield [x for x in inner(item)]
            elif next_item == ']':
                return
            else:
                yield next_item
    return list(next(inner(iter(string))))

def set_default_feature_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns_to_impute = ['guests', 'beds', 'bathrooms', 'bedrooms']
    for column in columns_to_impute:
        dataframe.loc[:, column] = dataframe[column].fillna(1)
    return dataframe

def clean_tabular_data(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = combine_description_strings(dataframe = dataframe, column = 'Description')
    dataframe = set_default_feature_values(dataframe)
    return dataframe

def load_airbnb(dataframe: pd.DataFrame, label: str) -> tuple:
    dataframe = dataframe.select_dtypes(exclude = ['object'])
    return tuple([dataframe.drop([label], axis = 1), dataframe[label].to_list()])        

if __name__ == '__main__':
    df = pd.read_csv('data\\AirBnbData.csv')
    df = clean_tabular_data(df, column = 'Description')
    df.to_csv('data\\clean_tabular_data.csv', index = False)
