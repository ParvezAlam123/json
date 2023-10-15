import json 
file = open("user.json", "r")
x = file.read()
finaldata = json.loads(x) 

print(finaldata) 
