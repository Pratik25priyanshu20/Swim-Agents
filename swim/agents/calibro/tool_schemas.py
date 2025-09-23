from pydantic import BaseModel, Field

class FilterByLakeInput(BaseModel):
    file_name: str = Field(..., description="The name of the CSV file.")
    lake_name: str = Field(..., description="The name of the lake to filter.")

class FilterByDateRangeInput(BaseModel):
    file_name: str = Field(..., description="The name of the CSV file.")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format.")

class PlotIndexInput(BaseModel):
    file_name: str = Field(..., description="The name of the CSV file.")
    index_name: str = Field(..., description="The name of the index (e.g., turbidity_index, chlorophyll_index).")

class SummarizeQualityInput(BaseModel):
    file_name: str = Field(..., description="The name of the CSV file.")