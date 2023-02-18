import os
def make_logging_client():
    cred_filename = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_filename:
        return None
    print("cred filename:", cred_filename)
    cred_string = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_STRING')
    print("cred string:", bool(cred_string))
    if not os.path.exists(cred_filename):
        if cred_string:
            print(f"writing cred string to {cred_filename}")
            with open(cred_filename, 'w') as f:
                f.write(cred_string)
        else:
            return None
    from google.cloud import logging
    logging_client = logging.Client()
    logging_client.setup_logging()
    return logging_client

logging_client = make_logging_client()
