# Greater Bay Area: Travel Choice API 🚄
A production-ready predictive API built with **FastAPI** and **scikit-learn** that provides data-driven advice for cross-border travel between Hong Kong and Mainland China.

## The Business Problem
Transportation hubs need to predict commuter choices to optimize network capacity. This API uses a Discrete Choice Model (L1 Logistic Regression) trained on real-world Greater Bay Area survey data to predict whether a commuter will choose Bus/MTR, High-Speed Rail, Taxi, Private Car, or eVTOL. All data processed through the API remains strictly confidential. 

*(Note: Data for long-distance work-related travel is currently limited and excluded from predictions).*

## How It Works
The model calculates the **Value of Travel Time Savings (VTTS)** and dynamically adjusts predictions based on:
* **Distance:** Short (50km), Medium (100km), Long (150km)
* **Trip Purpose:** Work vs. Non-Work
* **Hidden Costs & Preferences:** Evaluates in-vehicle time, crowding levels, transfer time, and customs clearance.

## 🚀 Endpoints
* **`GET /vtts`**: Instantly retrieves a static economic matrix of travel time valuation (HKD/Hour) across different distances based on trip purpose.
* **`POST /predict_choice`**: A dynamic simulation tool. Input custom parameters (fare, travel time, crowding level) to receive a precise probability score of a commuter choosing a specific transport mode.

## Tech Stack
* **Machine Learning:** Python, Pandas, Scikit-Learn 
* **Backend:** FastAPI, Pydantic, Uvicorn

## Local Testing
Due to privacy constraints, the original survey data is not included in this repository. To run this API locally, you must place your own `GBA Final Data (1).xlsx` (or a mock `sample_data.xlsx`) in the root directory.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/Greater-Bay-Area-Travel-Choice-Recommender.git](https://github.com/yourusername/Greater-Bay-Area-Travel-Choice-Recommender.git)
cd Greater-Bay-Area-Travel-Choice-Recommender
```

**2. Create a virtual environment**
*   **Windows:** *(If `python` opens the Windows Store, use `py`)*
    ```bash
    py -m venv venv
    .\venv\Scripts\activate
    ```
*   **Mac/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the server**
```bash
uvicorn api:app --reload
```

**5. Access the API**
Navigate to `http://127.0.0.1:8000/docs` to test the interactive Swagger UI.
