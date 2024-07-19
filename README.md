# Project Description
This project consists of **main 3 parts**:

## Fine-Tunining Model

 1. **Dataset Preparation**

	- main dataset used = [merge_dataset](https://huggingface.co/datasets/bbookarisara/merge_dataset)

> **merge_dataset** is created from merging between [*japanese_dataset*](https://huggingface.co/datasets/bbookarisara/japanese_dataset) and [*PDF_japanese_dataset*](https://huggingface.co/datasets/bbookarisara/japanese_fromPDF_dataset)
 2. **Fine-Tuning model**
 
	 - fine-tuning model =
   [bbookarisara/new-llama-3-8b-japanese-arisara](https://huggingface.co/bbookarisara/new-llama-3-8b-japanese-arisara)
> **pretrained_model** = *haqishen/Llama-3-8B-Japanese-Instruct*


**colab notebook link for further study** :
 - [DataMaker](https://colab.research.google.com/drive/1QDjrvao2fI0NfvN_GyDnf39QeGJglFJh?usp=sharing)
 - [PDF_DataMaker](https://colab.research.google.com/drive/1aFkK1nSwW0YRDZUgiJEbRbQveymNvQL1?usp=sharing)
 - [Fine-Tuning Model](https://colab.research.google.com/drive/17FsRHlL8DenRD5tBcGifLuSBVytJlT8A?usp=sharing) - ***T4 GPU runtime***

***Your personal HuggingFace_Token and WANDB_API_KEY are needed for running the notebooks***
   
## Database Monitoring
**Database** = [ChromaDB](https://docs.trychroma.com/getting-started)
```bash
pip install chromadb
```
browse database locally through [DB Browser for SQLite](https://sqlitebrowser.org/)

*Demonstration*

> After running **Chatbot Application**, you can monitor database through **DB Browser for SQLite** as following example figures.

## Deploy Chatbot Application

1. Download all files on [My Repository](https://github.com/bbookarisara/chatbot/)

>or use command below
```bash
git clone https://github.com/bbookarisara/chatbot.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. run app.py to deploy **Gradio Chatbot Application**
```bash
python3 app.py 
```

> running app.py will automatically load model through model.py

    

 ***with this parameter setting **'share=True'**, 
 you will get local url for private use and public url for sharing your application***
 
  ```bash
  demo.queue(max_size=20).launch(share=True)
```
![Screenshot from 2024-07-19 19-18-05](https://github.com/user-attachments/assets/3a88799d-75d7-4b2a-a7d0-ab417d4dac72)
