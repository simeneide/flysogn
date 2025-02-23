#%%

def main():
    import psycopg2
    import os
    uri = f'postgres://{st.secrets["aiven_user"]}:{st.secrets["aiven_password"]}@pg-weather-pg-weather.b.aivencloud.com:20910/defaultdb?sslmode=require'
    conn = psycopg2.connect(uri)

    query_sql = 'SELECT VERSION()'

    cur = conn.cursor()
    cur.execute(query_sql)

    version = cur.fetchone()[0]
    print(version)

# Write table
import polars as pl
import streamlit as st
st.secrets
import os
class Database:
    """ Simple wrapper around polars to read and write to aiven database """

    uri = f'postgres://{os.environ['AIVEN_USER']}:{os.environ['AIVEN_PASSWORD']}@pg-weather-pg-weather.b.aivencloud.com:20910/defaultdb?sslmode=require'
    def __init__(self):
        pass

    def read(self, query):
        df = pl.read_database_uri(query=query, uri=self.uri, engine="adbc")
        return df

    def write(self, df : pl.DataFrame, table_name):
        df.write_database(table_name=table_name, connection=self.uri, engine="adbc", if_table_exists="replace")
        return True

if __name__ == "__main__":
    db = Database()
    db.read("SELECT * FROM records")
    db.write(pl.DataFrame({"bs": [1, 2, 4], "kake" : ["hei","på","deg"]}), "records")
# %%