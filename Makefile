run-app:
	docker build -t graph-rec-app .
	docker run --rm -p 8501:8501 graph-rec-app