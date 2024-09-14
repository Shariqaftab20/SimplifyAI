FROM python:3.10-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV OPENAI_API_KEY="sk-proj-iwcVmbqDyiS7JGniy3bnT3BlbkFJFJVwJAr6frsYVwXn1SWK"
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
