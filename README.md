### Visual Language Semantic Search Engine

A simple semantic search application that takes in a text string and images, passes it through a Visual Language Model to generate text and image-embeddings and provides the most similar image from the image embeddings using K-nearest neighbours search.


Cool things about this project : 

- Switchable Visual Language Model Encoders via hf transformers. Currently Supporting: 
    - **All CLiP Versions**: Tested : ["openai/clip-vit-base-patch32"](https://huggingface.co/openai/clip-vit-base-patch32) , ["openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) , 
    - **All BLiP Versions**: Tested : ["Salesforce/blip-image-captioning-base"](), ["Salesforce/blip-image-captioning-large"](), 

    - Supports resource constrained systems by using quantized versions of CLiP via [clip.cpp](https://github.com/monatis/clip.cpp)

Work In Progress

- Fast Vector Search with [FAISS](https://github.com/facebookresearch/faiss)

### Environment Setup 

You would need a conda environment to install the dependencies
```shell
sudo apt install miniconda
```

Create new conda environment and install dependencies
```shell
conda env create --name vlss --file=environment.yml
conda activate vlss
```

### Run Streamlit Demo 

```shell
streamlit run demo.py
```


### Acknowledgement

This project is inspired from my older project on [visual place recognition](https://github.com/baj31415/visual-place-recognition). 
This project wouldn't have been possible without the existence of the following open-source libraries:

- [CLiP](https://github.com/openai/CLIP)
- [BLiP](https://github.com/salesforce/LAVIS)
- [clip.cpp](https://github.com/monatis/clip.cpp)
- [FAISS](https://github.com/facebookresearch/faiss)