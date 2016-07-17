"""
    Experimenting to offload computing to AzureML (WIP)
"""

import os

import bimbo_inventory_demand.secrets
import pandas as pd
from azureml import services

import bimbo_inventory_demand.python_bimbo.util_azure as uaz

# FILL these according to your azureml account setup
WORKSPACE_ID = bimbo_inventory_demand.secrets.WORKSPACE_ID
AUTH_TOKEN = bimbo_inventory_demand.secrets.AUTH_TOKEN


def azureml_main(frame1):
    return frame1


@services.types(a = float, b = float)
@services.returns(float)
def my_func(a, b):
    return a / b


@services.dataframe_service(a = int, b = int)
@services.returns(int)
def myfunc(df):
    return pd.DataFrame([df['a'][i] + df['b'][i] for i in range(df.shape[0])])


def main():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(curr_dir, 'data')
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    test_zip_path = os.path.join(data_path, 'test.csv.zip')
    client_data_path = os.path.join(data_path, 'cliente_tabla.csv')
    product_data_path = os.path.join(data_path, 'producto_tabla.csv')
    town_state_path = os.path.join(data_path, 'town_state.csv')
    print('Loading workspace...')
    ws = uaz.load_workspace(wid=WORKSPACE_ID, token=AUTH_TOKEN)
    uaz.print_user_ds(ws)
    print('Loading test dateset...')
    df_test = pd.read_csv(test_path)
    #uaz.upload_df_to_azure(ws, df_test, df_name='bim_test_ds', df_description='A pandas dataframe')
    #uaz.upload_file(ws, test_zip_path, 'bimbo_test')
    # uaz.upload_azure_blob(account=secrets.AZZURE_ACCOUNT_ID, account_key=secrets.STORAGE_ACCESS_KEY,
    #                       container=secrets.AZZURE_CONTAINER, blockblob='myblob', file=test_path, file_type='file/csv')

    print('Finished')


if __name__ == '__main__':
    main()