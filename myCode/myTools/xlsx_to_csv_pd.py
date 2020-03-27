import pandas as pd

# csv 与 csv 互转工具

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('suppodict.xlsx', index_col=0)
    data_xls.to_csv('suppodict.csv', encoding='utf-8')


if __name__ == '__main__':
    xlsx_to_csv_pd()