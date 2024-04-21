# Pluto.ai

Pluto.ai is a conversational Retrieval Augmented Generation (RAG) chatbot built with LangChain, Chainlit that uses Claude's capabilities for question-answering from documents.

![Screenshot (747)](https://github.com/aayushxrj/Pluto.ai/assets/111623667/2cec03e1-1016-440f-875b-1901b34f166c)

## Quick Start

### Clone the Repository

```
git clone https://github.com/aayushxrj/Pluto.ai
```

### Navigate to the Project Directory

```
cd ./Pluto.ai/
```

Now you can get the app up and running using either a [virtual environment](https://docs.python.org/3/library/venv.html) or [Docker](https://www.docker.com/).

### Set up a Virtual Environment

It's recommended to use a virtual environment to keep your project dependencies isolated. Follow these steps to set up a virtual environment:

1. Create a new virtual environment:

```
python -m venv venv
```

2. Activate the virtual environment:

#### On Windows
```
./venv/Scripts/activate
```
#### On Unix or Linux
```
source venv/bin/activate
```

### Install Dependencies
3. With your virtual environment activated, install the required dependencies using the following command:

```
pip install -r requirements.txt
```

### Deactivate the Virtual Environment
4. After installing the dependencies, you can deactivate the virtual environment:

```
deactivate
```

### Set up Anthropic API Key
Ensure you have an Anthropic API key to access the Claude LLM. You can obtain an API key from the [Anthropic Console](https://console.anthropic.com/dashboard).

5. Create a .env file in the project root directory and add the following line, replacing your_api_key with your actual API key:

```
ANTHROPIC_API_KEY=your_api_key
```

### Run the Application
6. Run the following script to start the Chainlit server:
```
chainlit run app.py -w
```

Interact with the application through the Chainlit interface. You can upload a file and ask questions related to the file's content.

### Docker Usage

If you have Docker installed and running, you can also run the application using Docker. Follow these steps:

1. Build the Docker image locally:

```
docker build -t aayushxrj/pluto.ai:latest
```
or directly pull from Docker Hub (recommended):

```
docker pull -t aayushxrj/pluto.ai:latest
```

2. Run the Docker container:

```
docker run --name pluto.ai -p 8000:8000 -e ANTHROPIC_API_KEY='your_api_key' aayushxrj/pluto.ai:latest
```

Replace `your_api_key` with your real Anthropic API key to gain access to the Claude LLM. You can obtain an API key from the [Anthropic Console](https://console.anthropic.com/dashboard).

You can access the app at `http://localhost:8000/`

## Usage

Once the Chainlit server is running, you'll be presented with an interface to upload a file.

![Screenshot (744)](https://github.com/aayushxrj/Pluto.ai/assets/111623667/b5ca305a-de88-475b-9db8-02119db42439)

After uploading a file, you can type your question in the text box and hit Enter.

![Screenshot (745)](https://github.com/aayushxrj/Pluto.ai/assets/111623667/0494108b-d8d3-4f4f-a779-48bad4b75dfd)


The application utilizes the Anthropic's API to access Claude to extract pertinent information from the uploaded file and furnish a response to your inquiry. It displays the source page, complete with page number and content, which serves as the basis for answering the question.

![Screenshot (746)](https://github.com/aayushxrj/Pluto.ai/assets/111623667/f3e8c33d-1a9b-4d14-988c-02700e51d466)

Thanks to Chainlit, it unveils the entire chain of thought that the RAG pipeline utilizes to arrive at the conclusive output at the end.

![screencapture-localhost-8000-2024-04-14-08_57_00](https://github.com/aayushxrj/Pluto.ai/assets/111623667/f13ad146-0222-4d74-a5ec-004cb8ee9ad7)

You can continue asking follow-up questions or upload a new file as desired.

## Contributing

If you wish to contribute to Project Pluto.ai, please adhere to the following steps:

- **Fork the repository**: Create a new branch (`git checkout -b feature/your-feature-name`)
- **Commit your changes**: Make your modifications (`git commit -m 'Add some feature'`)
- **Push to the branch**: Upload your changes (`git push origin feature/your-feature-name`)
- **Create a new Pull Request**: Submit your alterations for review

## License

Project Pluto.ai is licensed under the MIT License.

