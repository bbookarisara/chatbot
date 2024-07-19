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
 - [PDF_DataMaker](https://colab.research.google.com/drive/1aFkK1nSwW0YRDZUgiJEbRbQveymNvQL1?usp=sharing) - ***PDF files is required***
 - [Fine-Tuning Model](https://colab.research.google.com/drive/17FsRHlL8DenRD5tBcGifLuSBVytJlT8A?usp=sharing) - ***T4 GPU runtime***

***Your personal Hugging_Face_Token and WANDB_API_KEY are needed for running the notebooks***
   
## Database Monitoring
**Database** = [ChromaDB](https://docs.trychroma.com/getting-started)
```bash
pip install chromadb
```
browse database locally through [DB Browser for SQLite](https://sqlitebrowser.org/)

*Demonstration*

> After running **Chatbot Application**, you can monitor database through **DB Browser for SQLite** as following example figures.

monitor 2 processes through checking the terminal (It will display checkpoints for both processes!)
![Screenshot from 2024-07-19 19-46-34](https://github.com/user-attachments/assets/41bc3918-8a68-4782-9fc1-a69125f7d161)

test for chat application
![Screenshot from 2024-07-19 19-43-32](https://github.com/user-attachments/assets/b857a6e1-1edf-47a3-a4b6-857565878a62)

test for database
![Screenshot from 2024-07-19 19-44-36](https://github.com/user-attachments/assets/1ea505d6-b559-4203-8b3e-47a763d22029)

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
![Screenshot from 2024-07-19 19-42-17](https://github.com/user-attachments/assets/8e09684a-9a49-4de8-837e-974d6bc0e8dc)
