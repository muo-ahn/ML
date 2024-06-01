# ML
This is a hosting Summarization model on FastAPI.

## May 2024
Goal
- Web Crawling
- Summarization

## Installation

1. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the application

Start the FastAPI server using Uvicorn:
```sh
uvicorn app.main:app --reload
