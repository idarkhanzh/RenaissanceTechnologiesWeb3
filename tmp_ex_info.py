import requests
BASE_URL = "https://mock-api.roostoo.com"
r = requests.get(BASE_URL+"/v3/exchangeInfo")
print(r.status_code)
print(r.text[:2000])
