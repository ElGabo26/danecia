from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine, text
import os
import pymysql
from pandas import read_sql



class DataService:
    # Init class
    def __init__(self):
        load_dotenv()
        self.__host = os.getenv('SS_HOST')
        self.__user = os.getenv('SS_USER')
        self.__password = os.getenv('SS_PASS')
        self.__database = os.getenv('SS_NAME')
        self.__port = int(os.getenv('SS_PORT'))
    
    # Get data from database
    def get_data(self, query):
        engine = create_engine(
            f"mysql+pymysql://{self.__user}:{self.__password}@{self.__host}/{self.__database}"
        )
        try:
            with engine.connect() as connection:
                result = read_sql(query, connection)
            return result
        except Exception as e:
            return e
        finally:
            engine.dispose()
    def up_data(self,data,tabla):
        try:
            # Crear la conexión con SQLAlchemy
            engine = create_engine(f'mysql+pymysql://{self.__user}:{self.__password}@{self.__host}/DDM_GD')
            print(engine)
            
            # Cargar el DataFrame en la base de datos
            data.to_sql(name=tabla, con=engine, if_exists='append', index=False)
            
            print("Datos cargados exitosamente en la tabla", tabla)
        except Exception as e:
            print("Error al cargar datos:", e)
        finally:
            engine.dispose()
    def delete_data(self,tabla,condition):
        query=f" DELETE FROM {tabla} WHERE {condition}"
        try:
            # Crear la conexión con SQLAlchemy
            engine = create_engine(f'mysql+pymysql://{self.__user}:{self.__password}@{self.__host}/DDM_GD')
            with engine.begin() as conn:
                conn.execute(text(query))
            print(f"REGISTROS  BORRADOS  DE LA  TABLA {tabla}")
        except Exception as e:
            print(f"Error  al  borrar la   tabla {tabla}:",e)    




