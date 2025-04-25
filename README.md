# Agentic RAG System

A modular Retrieval-Augmented Generation (RAG) system that combines local document knowledge with web search capabilities to provide comprehensive answers to user queries.

## Features

- Document loading and processing (PDF support)
- Vector-based semantic search
- Local knowledge base
- Web search and scraping capabilities
- LLM-powered answer generation
- Error handling and logging
- Modular architecture

## System Workflow

![image](https://github.com/user-attachments/assets/aa336be8-0382-4d0d-9862-a5033a0ae94a)

The system follows a sophisticated decision-making process:

1. **Query Analysis**: When a user query is received, the system first evaluates if it can be answered using local knowledge.

2. **Knowledge Routing**:

   - If the query can be answered locally (Yes path), it pulls information from the local knowledge base
   - If not (No path), it retrieves data from external sources

3. **Information Sources**:

   - **Local Documents**: Vectorized and indexed document storage
   - **Internet Search**: Real-time web scraping and search capabilities

4. **Context Building**: The system combines and processes the retrieved information to build a comprehensive context.

5. **Answer Generation**: Finally, it uses an LLM to generate a precise, contextually relevant answer.

## Project Structure

```
.
├── config/
│   └── config.py           # Configuration settings
├── data/                   # Data directory
├── src/
│   ├── app.py             # Main application
│   ├── data_loader.py     # Document loading
│   ├── vector_store.py    # Vector store operations
│   ├── llm_interface.py   # LLM interactions
│   └── agents.py          # Web search agents
├── tests/                 # Test directory
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

## Usage

1. Place your documents in the `data/` directory
2. Run the application:
   ```bash
   python -m src.app
   ```

## Configuration

The system can be configured through `config/config.py`:

- Model settings (embedding model, LLM models)
- Vector store settings (chunk size, overlap)
- LLM parameters (temperature, max tokens)
- Error messages

## Error Handling

The system includes comprehensive error handling for:

- Missing API keys
- File not found errors
- Invalid PDF files
- Vector store creation errors
- LLM interaction errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
