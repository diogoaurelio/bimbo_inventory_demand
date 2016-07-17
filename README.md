

# Env requirements

## Setup xgboost
Setup [xgboost library](https://github.com/dmlc/xgboost/blob/master/doc/build.md#python-package-installation) (Extreme Gradient Boosting).

## Setup python
Make also sure you have all the required py dependencies:
```
pip install -r requirements.txt
```

# Usage

## Credentials
Two options:
- Create in the root of this repo a file called secrets and add the credentials (see [here](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-python-data-access/) for details how to get the credentials)
- Add credentials to ~/.azureml/settings.ini`as explained [here](https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python#specify-workspace-via-config):
```
[workspace]
id=4c29e1adeba2e5a7cbeb0e4f4adfb4df
authorization_token=f4f3ade2c6aefdb1afb043cd8bcf3daf
api_endpoint=https://studio.azureml.net
management_endpoint=https://management.azureml.net
``

## Getting started with python AzureML client
Browse [here](https://azure.microsoft.com/en-us/documentation/articles/machine-learning-python-data-access/) to see how to get credentials, and understand getting started.

## More about azureml py client
Browse [here](https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python).