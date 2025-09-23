class DataValidator:
    def validate(self, df):
        df['quality_score'] = 1.0
        df['validation_flags'] = ''
        return self._validate_required(df)

    def _validate_required(self, df):
        for col in ['station_id', 'latitude', 'longitude', 'geometry']:
            if col in df.columns:
                missing = df[col].isna()
                df.loc[missing, 'quality_score'] *= 0.6
                df.loc[missing, 'validation_flags'] += f'MISSING_{col.upper()};'
        return df