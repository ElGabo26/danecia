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
        if data.shape[0]==0:
            return "Empty dataSet"
        return data
    except Exception as  e:
        return e