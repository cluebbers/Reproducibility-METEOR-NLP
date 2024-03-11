build:
	@docker build . -t roguescores

start:
	@docker run -it --rm \
		-v /media/sf_roguescores/roguescores:/roguescores -w /roguescores \
		--name roguescores roguescores \
		ipython --no-banner -i main.py

clean:
	@docker rmi roguescores
