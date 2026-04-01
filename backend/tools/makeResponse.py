from tools.makeContext1 import build_prompt
from time import time
def getResponse(pregunta,client,modelName,temperature:float):
    print('|',end="|")
    system_instruction = build_prompt(pregunta)
    try:
        
        t0=time()    
        response = client.chat.completions.create(
        model=modelName,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": pregunta}
        ],
        temperature=temperature,
    )
        r=response.choices[0].message.content
        t1=time()
        
    except Exception as e:
        print(e)
        t1, t0=0,0
        print(e)
        r="DESCRIBE DDM_ERP.DIM_UNIDAD_NEGOCIO ;"
    print(t1-t0)
    return r
