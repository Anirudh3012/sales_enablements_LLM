import boto3
from botocore.exceptions import ClientError

class S3Utils:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def create_bucket_if_not_exists(self, bucket_name, region=None):
        """
        Create an S3 bucket in a specified region if it doesn't already exist.
        If a region is not specified, the bucket is created in the S3 default
        region (us-east-1).
        """
        try:
            if region is None:
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                location = {'LocationConstraint': region}
                self.s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
            print(f"Bucket '{bucket_name}' created successfully.")
        except ClientError as e:
            print(f"Error creating bucket: {e}")

    def push_file_to_bucket(self, file_name, bucket_name, object_name=None):
        """
        Upload a file to an S3 bucket.
        :param file_name: File to upload.
        :param bucket_name: Bucket to upload to.
        :param object_name: S3 object name. If not specified, file_name is used.
        """
        if object_name is None:
            object_name = file_name

        try:
            self.s3_client.upload_file(file_name, bucket_name, object_name)
            print(f"File '{file_name}' uploaded to '{bucket_name}' as '{object_name}'.")
        except ClientError as e:
            print(f"Error uploading file: {e}")

    def fetch_file_from_bucket(self, bucket_name, object_name, file_name=None):
        """
        Fetch a file from an S3 bucket.
        :param bucket_name: Name of the bucket.
        :param object_name: S3 object name.
        :param file_name: Filename to save the file as locally. If not specified, object_name is used.
        """
        if file_name is None:
            file_name = object_name

        try:
            self.s3_client.download_file(bucket_name, object_name, file_name)
            print(f"File '{object_name}' downloaded from '{bucket_name}' as '{file_name}'.")
        except ClientError as e:
            print(f"Error downloading file: {e}")
