# PDF Question Answering (RAG) App 🚀

This web application allows you to upload a PDF document, perform text extraction, and interact with a powerful Question Answering model based on Retrieval-Augmented Generation (RAG). The system utilizes the LangChain framework, Google Generative AI, and FAISS to process the PDF and generate intelligent responses to user queries.

---

## Features

- **PDF Upload**: Upload any PDF document and the system processes it for querying.
- **Question Answering**: Ask questions based on the uploaded document, and get answers derived from the document’s content.
- **FAISS Integration**: Efficient document retrieval using FAISS for similarity-based search.
- **Generative AI**: Leverages Google’s Generative AI to generate contextual responses to your queries.
- **Streamlit Interface**: A simple and interactive web interface for easy usage.

---

## Requirements

This project is built using **Python 3.10** and requires the following libraries:

- **langchain**
- **langchain_community**
- **langchain-google-genai**
- **python-dotenv**
- **streamlit**
- **langchain_experimental**
- **sentence-transformers**
- **langchain_faiss**
- **langchainhub**
- **pypdf**
- **rapidocr-onnxruntime**

You can install all the dependencies listed in the `requirements.txt` file.

---

## Setup Instructions

1. **Clone the repository**:
   
   ```bash
   [git clone https://github.com/yourusername/pdf-question-answering.git]
   cd Folder_name
   ```

2. **Create a virtual environment** (if not already done):
   
   ```bash
   python3.10 -m venv web1
   source web1/bin/activate   # On Windows: web1\Scripts\activate
   ```

3. **Install required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create a `.env` file in the root directory of the project and add your **Google API Key**:
   
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Run the application**:
   
   Start the Streamlit app using the following command:
   
   ```bash
   streamlit run app.py
   ```

   This will open the app in your default web browser, where you can upload PDFs and ask questions.

---

## How to Use the App

1. **Upload a PDF File**: 
   - Click on the **Upload** button to select a PDF document from your local system.
   
2. **Ask Questions**:
   - Once the PDF is processed, type your question in the text input box and click **Submit Question**.
   - The system will retrieve relevant information from the document and provide an answer.

3. **Adjust Temperature**:
   - You can adjust the **temperature** slider to control the randomness of the generative AI responses.

---

## Example Workflow

1. Upload a PDF document.
2. Wait for the document to be processed and indexed.
3. Enter a question related to the PDF content.
4. Get an accurate and concise answer based on the document's context.

---

## Project Structure

```
pdf-question-answering/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # List of Python dependencies
├── .env                  # Environment file to store API keys
├── README.md             # Project documentation
├── images/               # (Optional) Folder for assets like icons or logos
└── temp/                 # Temporary folder used during file upload processing
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

- **Google API Key**: Make sure your Google API key is correctly set in the `.env` file.
- **Dependencies**: Ensure all dependencies are installed in your virtual environment.
- **Streamlit Errors**: If you encounter issues running Streamlit, try reinstalling Streamlit or clearing your browser cache.

---

## Acknowledgments

- **LangChain**: For providing the framework for building language-based chains.
- **Google Generative AI**: For providing powerful AI models.
- **FAISS**: For efficient similarity search and vector indexing.
- **Streamlit**: For creating the simple and interactive web interface.

---

Feel free to reach out or contribute to the repository. Happy coding! ✨

---

### Example `requirements.txt`

```plaintext
langchain==0.1.0
langchain_community==0.1.0
langchain-google-genai==0.1.0
python-dotenv==0.21.0
streamlit==1.15.0
langchain_experimental==0.1.0
sentence-transformers==2.2.0
langchain_faiss==0.1.0
langchainhub==0.0.7
pypdf==3.0.0
rapidocr-onnxruntime==0.3.0
```
