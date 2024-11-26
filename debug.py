import pandas as pd
min_date = pd.to_datetime("2020-01-01")
date = min_date + pd.Timedelta(days=1,unit = 'days')
date = date.strftime('%Y-%m-%d')
print(date)