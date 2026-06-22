# Greater-Bay-Area-Travel-Choice-Recommender
A model that provide advice for people deciding how they travel crossborder(from hong kong to mainland china),the api will give advice base on users preference on hidden cost, distance of travelling and income.All data inputed will be confidential and use for the advice only (for work purpose long distance travel, I dont have enough data now and it is unavaliable)
## How It Works
The model calculates the **Value of Travel Time Savings (VTTS)**. It dynamically adjusts predictions based on:
* **Distance:** Short (50km), Medium (100km), Long (150km)
* **Trip Purpose:** Work vs. Leisure
* **Traveler Constraints:** E.g., Does the user hold a valid driving license?

## Tech Stack
* **Machine Learning:** Python, pandas, scikit-learn (Logistic Regression with L1 penalty)
* **Backend:** FastAPI, pydantic, uvicorn
* **Deployment:** (You will fill this in later when we put it on a cloud server)

## Local Testing
To run this API on your local machine:
1. Clone the repository.
2. Run `pip install -r requirements.txt`.
3. Start the server: `uvicorn main:app --reload`.
4. Navigate to `http://127.0.0.1:8000/docs` to test the interactive Swagger UI.
