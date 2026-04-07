from tools.DataService import DataService
import re

def getData(service:DataService,result):
    patron = r"SELECT.*?;"
    resultado = re.search(patron, result, re.DOTALL | re.IGNORECASE)

    if resultado:
        query=resultado.group()
    else:
        query=None
    try:
        data=service.get_data(query)
        return data
    except Exception as  e:
        print(type(e))
        return e