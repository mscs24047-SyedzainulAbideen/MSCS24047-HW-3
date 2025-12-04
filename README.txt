README: Information Retrieval System (Boolean + TF-IDF + BM25)
Overview
This project implements a local Information Retrieval (IR) system capable of retrieving documents using three methods: - Boolean Retrieval: Exact match of all query terms. - TF-IDF Retrieval: Ranking documents based on cosine similarity of TF-IDF vectors. - BM25 Retrieval: Probabilistic relevance ranking using the BM25 algorithm.
The system is designed to work on a CSV dataset containing text documents.
Project Structure
ir_system/
├─ data/
│  └─ Articles.csv          # Your dataset
├─ src/
│  ├─ preprocess.py         # Text cleaning and tokenization
│  ├─ retriever.py          # Boolean, TF-IDF, BM25 retrieval
│  ├─ main.py               # Run sample queries
├─ requirements.txt         # Python dependencies
└─ README.md                # This file
Setup Instructions
1. Prerequisites
•	Python 3.9+ installed
•	Pip package manager installed
2. Install Dependencies
Navigate to the project directory and run:
pip install -r requirements.txt
3. Dataset
Place your dataset Articles.csv inside the data/ folder. Ensure the dataset contains a text column such as Article, Text, Content, or Body.
4. Run the System
Navigate to the src folder and execute:
cd src
python main.py
The system will output the top 5 documents for a sample query using: - Boolean retrieval - TF-IDF retrieval - BM25 retrieval
5. Modify Query
To test your own query, open main.py and change the query variable:
query = "your custom query here"
6. Notes
•	The system automatically detects the best text column.
•	It handles multiple encodings when loading CSV files.
•	Stopwords are removed during preprocessing.
Dependencies
•	pandas
•	numpy
•	scikit-learn
•	nltk
•	rank_bm25
Contact
For issues or questions, please contact the developer.
