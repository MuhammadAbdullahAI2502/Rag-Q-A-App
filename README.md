# 🚀 Advanced RAG Q&A Assistant

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain that allows users to upload documents and ask questions in any language.

## ✨ Features

- 🌍 **Multi-language support** - Ask questions in any language
- 📄 **Multiple file formats** - Support for PDF, DOCX, TXT files  
- 💾 **Chat history persistence** - Save and manage conversation history
- 🔍 **Intelligent document search** - Advanced retrieval with context awareness
- ⚡ **Fast responses** - Powered by Groq LLM
- 🎨 **Clean UI** - Modern and intuitive interface
- 🔒 **Secure** - API keys handled securely

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Key
- Go to [Groq Console](https://console.groq.com/)
- Sign up/Login
- Create new API key
- Copy the key (starts with `gsk_...`)

### 4. Run the application
```bash
streamlit run rag.py
```

### 5. Use the app
1. Enter your Groq API key in the sidebar
2. Upload your documents (PDF, DOCX, TXT)
3. Ask questions in any language!

## 🌐 Live Demo

Deploy on Streamlit Cloud: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-streamlit-url)

## 📋 Requirements

- Python 3.8+
- Groq API key
- Internet connection for LLM API calls

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Gemma2-9B-IT)
- **Embeddings**: SentenceTransformers
- **Vector Store**: DocArrayInMemorySearch
- **Document Processing**: LangChain
- **File Support**: PyPDF, python-docx

## 📁 Project Structure

```
Rag/
├── rag.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── .streamlit/
    └── config.toml    # Streamlit configuration
```

## 🔧 Configuration

The app supports various configuration options in the sidebar:
- API key input
- Model selection
- Advanced settings (chunk size, temperature, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [LangChain](https://langchain.com/) for RAG capabilities
- [Groq](https://groq.com/) for fast LLM inference
- [Hugging Face](https://huggingface.co/) for embeddings

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

Developed By Muhammad Abdullah
