import os
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient

def getBlobServiceClient():
    connect_string = "DefaultEndpointsProtocol=https;AccountName=dojdemostorage;AccountKey=A/SMTTpgoQnrc4E3fvgBUfIXcGoIveQxOEL439nuh26NIGOqFA8KU6zjcfR43z5Gua22O1DVdO/k3/Ad1YxgFg==;EndpointSuffix=core.windows.net"
    blobService = BlobServiceClient.from_connection_string(conn_str=connect_string)
    return blobService

def get_container():
    blobService = getBlobServiceClient()
    container_client = blobService.get_container_client("dojdemo")
    return container_client

def get_container_name():
    container_name = get_container()

def uploadblob():
    blobService = getBlobServiceClient()
    container_client = get_container()
    local_path = "C:\\Users\AzureUser\Documents\dojfiles"
    for files in os.listdir(local_path):
        blob_client = container_client.upload_blob(name=files,data=files)
        properties = blob_client.get_blob_properties()

if __name__== '__main__':
    blobService = getBlobServiceClient()
    container_client = get_container()
    uploadblob()
