from get_connection import get_db_connection

sample_data = {
    "location_name": "Lake Constance",
    "latitude": 47.654,
    "longitude": 9.479,
    "timestamp": "2023-07-15 10:00:00",
    "chlorophyll_a": 12.5,
    "turbidity": 8.3,
    "dissolved_oxygen": 7.1,
    "ph": 8.0,
    "conductivity": 250.0,
    "redox_potential": 300.0,
    "source": "UBA",
    "confidence_score": 0.92
}

query = """
INSERT INTO harmonized_measurements (
    location_name, latitude, longitude, timestamp, chlorophyll_a, turbidity,
    dissolved_oxygen, ph, conductivity, redox_potential, source, confidence_score
) VALUES (
    %(location_name)s, %(latitude)s, %(longitude)s, %(timestamp)s, %(chlorophyll_a)s, %(turbidity)s,
    %(dissolved_oxygen)s, %(ph)s, %(conductivity)s, %(redox_potential)s, %(source)s, %(confidence_score)s
);
"""

conn = get_db_connection()
cur = conn.cursor()
cur.execute(query, sample_data)
conn.commit()

print("✅ Sample data inserted successfully.")

cur.close()
conn.close()