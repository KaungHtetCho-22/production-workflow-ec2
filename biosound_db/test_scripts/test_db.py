from sqlalchemy import create_engine

DATABASE_URL = "mysql+mysqlconnector://biosound_user:biosound_password@localhost/species_db"
engine = create_engine(DATABASE_URL)

# Test the connection
try:
    connection = engine.connect()
    print("Database connection successful")
    connection.close()
except Exception as e:
    print(f"Error connecting to the database: {e}")
