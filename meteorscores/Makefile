build:
	@docker build . -t meteorscores

start:
	@docker run -it --rm \
		-v /media/sf_meteorscores/meteorscores:/meteorscores -w /meteorscores \
		--name meteorscores meteorscores \
		ipython --no-banner -i main.py

clean:
	@docker rmi meteorscores
