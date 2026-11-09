# Audit Workbench with Persistent History

## Run

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

## What the new History does

The app creates `audit_history.db` automatically in the same folder.

It records:

- file import date/time
- staff name
- filename
- sheet name
- header row
- population / number of records
- method used
- selected row count
- systematic sample size, interval, random start and seed
- Benford selected digits
- weekend/public-holiday/round-million test settings

Use the **History** page in the left sidebar to review or download the history as CSV.

The uploaded source file is not changed by the tests.
