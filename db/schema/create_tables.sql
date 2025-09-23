CREATE TABLE IF NOT EXISTS harmonized_measurements (
    id SERIAL PRIMARY KEY,
    location_name TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    timestamp TIMESTAMP,
    chlorophyll_a DOUBLE PRECISION,
    turbidity DOUBLE PRECISION,
    dissolved_oxygen DOUBLE PRECISION,
    ph DOUBLE PRECISION,
    conductivity DOUBLE PRECISION,
    redox_potential DOUBLE PRECISION,
    source TEXT,
    confidence_score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);