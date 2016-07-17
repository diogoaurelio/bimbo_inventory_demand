from azureml import Workspace
from azureml import DataTypeIds
from azureml import AzureMLConflictHttpError
from azureml import services
from azure.storage.blob import BlobService

import bimbo_inventory_demand.secrets  # just a py file w AzureML credentials..


def load_workspace(wid=None, token=None, endpoint=None):

    # Either load config from ~/.azureml/settings.ini OR
    # pass as params
    if all((wid, token)):
        params = dict(workspace_id=wid, authorization_token=token)
        if endpoint:
            params['endpoint'] = endpoint
        ws = Workspace(**params)
    else:
        ws = Workspace()
    return ws


def upload_file_azureml(ws, ws_id, ws_token, file_path, azure_file_name, endpoint=None):
    """
    Note: this func only works for small datasets...
    :param ws:
    :param df:
    :param df_name:
    :param df_description:
    :return:
    """
    @services.attach((file_path, azure_file_name))
    def f(): pass
    try:
        attachment = services.attach((file_path, azure_file_name))(f)
        result = services.publish(attachment, ws_id, ws_token)
        #result = services.publish(f, WORKSPACE_ID, AUTH_TOKEN)
        print(result)
        print_user_ds(ws)
    except AzureMLConflictHttpError as err:
        msg = 'AzureML is complaining that Dataset already exists: {0}'\
            .format(err)
        print(msg)
    except Exception as err:
        msg = 'Error while trying to upload \'{0}\': {1}'\
            .format(file_path, err)
        print(msg)


def print_user_ds(ws):
    print('Current user datasets are:')
    for ds in ws.user_datasets:
        print('\t - {}'.format(ds.name))


def upload_azure_blob(account, account_key, container, filename, file, file_type='file/csv'):

    block_blob_service = BlobService(account_name=account, account_key=account_key)

    # block_blob_service.put_block_blob_from_path(
    #    container,
    #    blockblob,
    #    file,
    #    x_ms_blob_content_type='file/csv'
    # )
    block_blob_service.create_blob_from_stream(container, filename, file)
    generator = block_blob_service.list_blobs(container)
    for blob in generator:
        print(blob.name)


def upload_df_to_azure(ws, df, df_name, df_description='A pandas dataframe'):
    """
    Note: this func only works for small datasets...
    :param ws:
    :param df:
    :param df_name:
    :param df_description:
    :return:
    """
    try:
        result = ws.datasets.add_from_dataframe(
            dataframe=df,
            data_type_id=DataTypeIds.GenericCSV,
            name='my new dataset',
            description=df_description
        )
        print(result)
        print_user_ds(ws)

    except AzureMLConflictHttpError as err:
        msg = 'AzureML is complaining that Dataset already exists: [{0}]- {1}'\
            .format(err.status_code, err)
        print(msg)
    except Exception as err:
        msg = 'Error while trying to upload {0} dataframe: [{1}]- {2}'\
            .format(df_name, err.status_code, err)
        print(msg)

