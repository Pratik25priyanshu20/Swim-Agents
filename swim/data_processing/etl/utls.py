from dataclasses import dataclass

@dataclass
class LakeLocation:
    lake_id: str
    latitude: float
    longitude: float

@dataclass
class DataSource:
    name: str
    url: str