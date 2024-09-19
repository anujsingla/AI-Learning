# salesforce_api.py

import requests

domain = 'https://anujsingla-dev-ed.develop.my.salesforce.com'

def get_salesforce_access_token():
    username = '<username>'
    password = '<password>' + '<token>'
    ConsumerKey = '<consumerkey>'
    ConsumerSecret = '<consumersecret>'
    
    payload = {
        'grant_type': 'password',
        'client_id': ConsumerKey,
        'client_secret': ConsumerSecret,
        'username': username,
        'password': password   
    }

    oauth_endpoint = '/services/oauth2/token'
    response = requests.post(domain + oauth_endpoint, data=payload)

    if response.status_code == 200:
        access_token = response.json().get('access_token')
        return access_token
    else:
        raise Exception(f"Failed to get access token: {response.text}")

def get_salesforce_report_data(access_token, report_id):

    headers = {
        'Authorization': 'Bearer ' + access_token
    }

    report_endpoint = f'/services/data/v59.0/analytics/reports/{report_id}'
    # reportendpoint = '/services/data/v59.0/analytics/reports/00ONS000001Yvsv2AC'

    report_response = requests.get(domain + report_endpoint, headers=headers)

    if report_response.status_code == 200:
        return report_response.json()
    else:
        raise Exception(f"Failed to fetch report data: {report_response.text}")
    

def get_salesforce_adults_data(access_token):
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/json'
    }
    
    api_url = f"{domain}/services/data/v59.0/query/?q=SELECT+Id,+X39__c,+stategov__c,+X77516__c,+bachelors__c,+X13__c,+never_married__c,+Adm_clerical__c,+Not_in_family__c,+White__c,+Male__c,+X2174__c,+X0__c,+X40__c,+United_States__c,+X50K__c+FROM+Adult__c"

    all_records = []

    while api_url:
        response = requests.get(api_url, headers=headers)
        data = response.json()
        # print('data', data)
        if isinstance(data, dict) and 'records' in data:
            all_records.extend(data['records'])
        else:
            print(f"Unexpected response structure: {data}")
            break
        
        if 'nextRecordsUrl' in data:
            api_url = domain + data['nextRecordsUrl']
        else:
            api_url = None
    return all_records
