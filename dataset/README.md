# FUR-API DATASET

## List of columns for the dataset:
[timestamp,	agent,	method,	request_body,	request_headers,	request_url,	response_body,	response_headers,	response_size,	source_ip,	source_port,	status,	target_ip,	target_port,	response_time,	user_identity,	type]

## Basic features of each API request sample
```
timestamp: Time when the API request was made.  
agent: The client or software that initiated the API request.   
method: The type of HTTP request.  
request_body: Data sent by the client to the server in the API request.  
request_headers: Additional information sent by the client to the server along with the API request.  
request_url: Uniform Resource Locator that specifies the endpoint of the API being accessed.  
response_body: Data returned by the server in response to the API request.  
response_headers: Additional information sent by the server along with the API response.  
response_size: Size of the response sent by the server.  
source_ip: The IP address of the device or entity that initiated the API request.  
source_port: The port number on the source device that is used for the communication.   
status: The status code returned by the server indicates the outcome of the API request.  
target_ip: The IP address of the server or endpoint that received the API request.  
target_port: The port number on the target server where the API request is received and processed.  
response_time: Time taken by the server to process the API request and send back the response to the client.
user_identity: User identity for who initiated the request.
type: API request type.
```

## Dataset statistics 
|          | Normal Samples | Anomaly Samples | Anomaly Type |
| -------- | -------------- | --------------- | ------------ |
| Training set | 20000 | 60 | 4 |
| Test set | 1000 | 150 | 7 |

