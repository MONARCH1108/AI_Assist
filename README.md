# Research Assistant

A powerful AI-driven research tool that helps you gather, organize, and analyze information from multiple sources.

## Features

- **Multi-source Research**: Search and aggregate information from Wikipedia, arXiv, and web sources
- **Document Analysis**: Upload PDFs or provide URLs for in-depth analysis
- **Long-term Memory**: Store and retrieve research findings across sessions
- **Semantic Search**: Find relevant information using natural language queries
- **Article Summarization**: Automatically generate concise summaries of articles
- **Research Organization**: Organize findings by topic and generate structured research notes
- **User Preferences**: Customize research experience with preferred sources and citation styles

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features in Depth](#features-in-depth)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7+
- Flask
- LangChain
- Hugging Face Transformers
- Google Generative AI (Gemini)

### Setup

1. Clone the repository:
    
    bash
    
    ```bash
    git clone https://github.com/yourusername/research-assistant.git
    cd research-assistant
    ```
    
2. Install dependencies:
    
    bash
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Set up API keys:
    
    bash
    
    ```bash
    # Create a .env file with your API keys
    GEMINI_API_KEY=your_gemini_api_key
    SERPER_API_KEY=your_serper_api_key
    ```
    
4. Run the application:
    
    bash
    
    ```bash
    python app.py
    ```
    
5. Access the web interface at `http://localhost:5000`

## Usage

### Basic Research Query

Simply ask a question in natural language:

```
What are the latest developments in quantum computing?
```

The assistant will search multiple sources and provide a comprehensive answer with references.

### Document Analysis

1. **Analyze a PDF**:
    - Upload a PDF using the interface
    - Ask questions about the document's content
2. **Analyze a Web Page**:
    - Enter a URL in the input field
    - Start asking questions about the content

### Research Organization

Organize your findings with commands like:

```
Organize my research notes on quantum computing
```

or

```
Generate research notes on climate change
```

## Features in Depth

### Long-Term Memory

The application maintains a SQLite database (`memory.db`) that stores:

- Questions and answers
- Research topics
- Article summaries
- User preferences

This allows the assistant to remember your research interests and findings across sessions.

### Vector Search

Documents are processed using:

1. Document loading (PDF, web content)
2. Text splitting into manageable chunks
3. Embedding generation using Hugging Face models
4. Storage in a Chroma vector database
5. Semantic retrieval based on query relevance

### Research APIs

The application integrates with multiple research sources:

- **Wikipedia**: For encyclopedic knowledge
- **arXiv**: For scientific papers
- **Google (via Serper API)**: For web results

### AI Integration

The application uses Google's Gemini model to:

- Generate concise summaries
- Detect research topics
- Identify research gaps
- Synthesize information from multiple sources

## API Reference

### Endpoints

|Endpoint|Method|Description|
|---|---|---|
|`/`|GET|Main interface|
|`/process_wiki`|POST|Process a Wikipedia URL|
|`/process_pdf`|POST|Upload and process a PDF file|
|`/ask`|POST|Ask a question|
|`/clear`|POST|Clear current session|
|`/search_arxiv`|POST|Search arXiv papers|
|`/organize_notes`|POST|Generate organized research notes|
|`/topic_notes/<topic>`|GET|Get notes for a specific topic|
|`/save_preferences`|POST|Save user preferences|
|`/get_topics`|GET|Get user's research topics|
|`/summarize_url`|POST|Summarize article at URL|
|`/get_user_profile`|GET|Get user's research profile|

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for the document processing pipeline
- [Hugging Face](https://huggingface.co) for embedding models
- [Google Generative AI](https://ai.google.dev/) for the Gemini model
- [Flask](https://flask.palletsprojects.com/) for the web framework
