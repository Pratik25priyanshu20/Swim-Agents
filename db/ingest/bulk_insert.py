import psycopg2
from db.ingest.get_connection import get_db_connection

def bulk_insert_records(records):
    if not records:
        print("⚠️ No records to insert.")
        return

    query = """
    INSERT INTO harmonized_measurements (
        location_name, latitude, longitude, timestamp,
        chlorophyll_a, turbidity, dissolved_oxygen, ph,
        conductivity, redox_potential, source, confidence_score
    )
    VALUES (
        %(location_name)s, %(latitude)s, %(longitude)s, %(timestamp)s,
        %(chlorophyll_a)s, %(turbidity)s, %(dissolved_oxygen)s, %(ph)s,
        %(conductivity)s, %(redox_potential)s, %(source)s, %(confidence_score)s
    );
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        for record in records:
            try:
                cur.execute(query, record)
            except Exception as e:
                print(f"❌ Failed to insert record: {e}")
                continue

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Inserted {len(records)} records into the database.")

    except Exception as e:
        print(f"❌ Bulk insert failed: {e}")