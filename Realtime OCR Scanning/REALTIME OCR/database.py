import mysql.connector

def save_to_db(extracted_text):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",           # Replace with your MySQL username
        password="So@080903",   # Replace with your MySQL password
        database="ocr_db"      # The database you created earlier
    )

    cursor = connection.cursor()
    sql = "INSERT INTO ocr_texts (extracted_text) VALUES (%s)"
    cursor.execute(sql, (extracted_text,))
    connection.commit()
    cursor.close()
    connection.close()
