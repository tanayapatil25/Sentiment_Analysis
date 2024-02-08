import requests

url = 'http://127.0.0.1:5000/predict'  # Adjust the URL based on your server address

data = {
    'reviews': ['This is a positive review.', 'This is a negative review.']
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
