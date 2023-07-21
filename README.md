# PDF Search Engine with Streamlit, ChromaDB, and AWS SageMaker using jumpstart-dft-meta-textgeneration-llama-2-7b-f

This project is a simple search engine for PDF files. It uses Streamlit for the web interface, ChromaDB for storing and querying document embeddings, and AWS SageMaker for generating responses to user queries.


## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have an AWS account.
- You have installed the required Python packages listed in the `requirements.txt` file.

## AWS Setup

Before running the code, you need to set up an AWS SageMaker domain and user profile. Follow these steps:

1. Log in to your AWS account and navigate to Amazon SageMaker.
2. Create a new domain and user profile.
3. Navigate to the JumpStart model section and launch the `jumpstart-dft-meta-textgeneration-llama-2-7b-f` model.
4. Wait for the endpoint to be deployed. This may take a few minutes.

Once the endpoint is deployed, you can proceed with the next steps.


## Features

- Upload a PDF file and extract the text from each page.
- Store the text and its embeddings in ChromaDB.
- Enter a query and get the most relevant page from the PDF.
- Use AWS SageMaker to generate a response to the query.

## Installation

1. Clone this repository.
2. Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. Upload a PDF file.
2. Enter a query related to the content of the PDF.
3. Click "Submit Query" to get the most relevant response.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project uses the following license: [MIT No Attribution](LICENSE).