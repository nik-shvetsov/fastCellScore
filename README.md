# Fast TILs estimation in lung cancer WSIs based on semi-stochastic patch sampling

This repository hold the code for the publication: [Link to arxiv](https://arxiv.org/abs/2405.02913)

The easiest way to obtain the whole pipeline with the models is to download container from dockerhub:
```
docker pull nikshv/fastcellscore:1.2
docker run -it --gpus all nikshv/fastcellscore:1.2
```

There you can adjust parameters in pipeline.py (params dictionary) and run the pipeline with:
```
python pipeline.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
