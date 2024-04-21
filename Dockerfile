FROM python:3.11.4

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt  --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

ENV ANTHROPIC_API_KEY placeholder_value

CMD ["chainlit", "run", "app.py"]