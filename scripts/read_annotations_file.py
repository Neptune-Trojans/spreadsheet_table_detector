import argparse
import ast
import pandas as pd

from src.utils.dataframe_tools import init_dataframe_view
from src.utils.spreadsheet_reader import SpreadsheetReader

if __name__ == '__main__':
    init_dataframe_view()

    parser = argparse.ArgumentParser(description='Reading the annotations')
    parser.add_argument('--labels_file', type=str, help='table annotations')
    parser.add_argument('--spreadsheets_folder', type=str, help='output folder')
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_file)
    labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)

    spreadsheet_reader = SpreadsheetReader(300,300)
    tables, backgrounds = spreadsheet_reader.load_dataset_maps(labels_df, args.spreadsheets_folder)