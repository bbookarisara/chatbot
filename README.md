
# Project Description

This project consists of **three main parts**:

## Fine-Tuning Model

There are **two options** for fine-tuning the model:

1. **Traditional Method (Coding)**
2. **LLaMA Factory (GUI) - Preferred!**

For data used in both methods, a **DataMaker notebook** is provided [here](https://colab.research.google.com/drive/1mGV1MnFb7plmwFo_YB1HwgXIxF_7hwBo?usp=sharing).

### Data Sources

| Method                  | Dataset                                                                                |
|-------------------------|----------------------------------------------------------------------------------------|
| Traditional Method      | [merge_dataset](https://huggingface.co/datasets/bbookarisara/merge_dataset)            |
| LLaMA Factory Method    | spft_format.json                                                                       |

- **Note**:
  - For `merge_dataset`, **request access** is required due to the private repository on Hugging Face.
  - For `spft_format.json`, download it through the [DataMaker notebook](https://colab.research.google.com/drive/1mGV1MnFb7plmwFo_YB1HwgXIxF_7hwBo?usp=sharing).
  - **cbt_interview_text.pdf** is required to make fine-tuning data.

## Traditional Method

- **Model**: [bbookarisara/new-llama-3-8b-japanese-arisara](https://huggingface.co/bbookarisara/new-llama-3-8b-japanese-arisara)
- **Pretrained Model**: `haqishen/Llama-3-8B-Japanese-Instruct`

**Required**: Your personal Hugging Face Token and WANDB_API_KEY.

**Warnings**:
- **Change runtime** type to T4 GPU to run this [fine-tuning notebook](https://colab.research.google.com/drive/17FsRHlL8DenRD5tBcGifLuSBVytJlT8A?usp=sharing)
- *The traditional method* adds '**PeftModel**' in the 2 last snippet of the notebook; use `peft.AutoPeftModelForCausalLM` instead of `transformers.AutoModelForCausalLM` when loading.
- *The LLaMA Factory method* only loads model using `transformers.AutoModelForCausalLM` instead of   *peft.AutoPeftModelForCausalLM* and requires an `config.json`. instead of *adapter_config.json*

## LLaMA Factory

Set up by reading the instructions in the [README](https://github.com/hiyouga/LLaMA-Factory).

### Additional Modifications

- **Adding Custom Dataset**:
  Modify `dataset_info.json` as shown below. Ensure the format is correct according to the [official format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) to avoid errors.

> In this case, **Alpaca Forma**t is used for the supervised fine-tuning dataset.  

 1. format data from [DataMaker notebook](https://colab.research.google.com/drive/1mGV1MnFb7plmwFo_YB1HwgXIxF_7hwBo?usp=sharing)
2. import it to the `LLaMA-Factory/data`
3. update `LLaMA-Factory/data/dataset_info.json`

**Warnings**:
- Use `Fp16` instead of `bf16` if the training machine is not A100.
- For running on Colab or other servers, ensure `share=True` in `src/llamafactory/webui/interface.py`


### Running the Server

1.  **Navigate to the LLaMA Factory Directory**:
    
    `cd LLaMA-Factory` 
    
2.  **Start the Server**:
 
    `python3 /home/arisara/LLaMA-Factory/src/webui.py` 
    
3.  **Add Dataset and Adjust Parameters**:
    
    -   Add your **custom dataset** and **adjust parameters** in the training section.
    -  Monitor the loss function graph to see how the loss decreases with each epoch.
<img width="1266" alt="model_config" src="https://github.com/user-attachments/assets/d419ac32-3a8c-4784-bf3d-3d3d92ff4f9f">
<img width="1242" alt="traininig_config" src="https://github.com/user-attachments/assets/fad98ce8-d108-456b-9efa-0783e85e3a8d">
<img width="1213" alt="lora_config" src="https://github.com/user-attachments/assets/6c3df038-a918-4f19-a998-92784e67594a">

4.  **Evaluate and Predict**:
    
    -   Use the **same trained dataset** to evaluate the model and make predictions.
  <img width="1261" alt="eval" src="https://github.com/user-attachments/assets/38718442-8f3f-428e-82e9-086211ba06c5">

5.  **Interact with the Model**:
    
    -   Test the model by chatting with it. You can adjust parameters such as  `max_new_tokens`,  `top_p`,  `top_k`, and  `temperature`  to optimize performance.
  <img width="1235" alt="chat1" src="https://github.com/user-attachments/assets/5b7ada19-dc27-46c4-97d8-8a1cb4174ab4">

<img width="1228" alt="chat2" src="https://github.com/user-attachments/assets/f3e1a44a-a0a9-413f-8847-f1e9c5b62577">

6.  **Export the Model**  (Optional):
    
    -   Export the trained model to a *local directory* or push it to a *Hugging Face repository.*
    -   <img width="1242" alt="export" src="https://github.com/user-attachments/assets/f8dc223b-a109-4919-bae1-8da8389e47b2">

  
## Database Monitoring

**Database**:  [ChromaDB](https://docs.trychroma.com/getting-started)

Install:
```
pip install chromadb 
```
Browse the database locally with  [DB Browser for SQLite](https://sqlitebrowser.org/).

### Demonstration

After running the  **Chatbot Application**, monitor the database through  **DB Browser for SQLite**.

-   Monitoring processes through the terminal.
![Screenshot from 2024-07-19 19-46-34](https://github.com/user-attachments/assets/41bc3918-8a68-4782-9fc1-a69125f7d161)
-   Testing the chat application.
![Screenshot from 2024-07-19 19-43-32](https://github.com/user-attachments/assets/b857a6e1-1edf-47a3-a4b6-857565878a62)
-   Testing the database.
![Screenshot from 2024-07-19 19-44-36](https://github.com/user-attachments/assets/1ea505d6-b559-4203-8b3e-47a763d22029)

## Deploy Chatbot Application

1.  Download all files from  [My Repository](https://github.com/bbookarisara/chatbot/).
    
    Or use the command below:
    
    `git clone https://github.com/bbookarisara/chatbot.git` 
    
2.  Install dependencies:
    
    `pip install -r requirements.txt` 
    
3.  Run  `app.py`  to deploy the  **Gradio Chatbot Application**:
    
    `cd chatbot-main
    python3 app.py` 
    

Running  `app.py`  will automatically load the model through  `model.py`.

**Note**: With the parameter setting  `'share=True'`, you will get a local URL for private use and a **public URL** for sharing your application.

`demo.queue(max_size=20).launch(share=True)` 

![Screenshot from 2024-07-19 19-42-17](https://github.com/user-attachments/assets/8e09684a-9a49-4de8-837e-974d6bc0e8dc)

-   Running the application with the share parameter.

## Further Improvements

-   [Deploy the Gradio server on Nginx](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx) to serve multiple users.
-   Improve the training dataset by using more comprehensive data.
-   Leverage the **history database** using [vector stores](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/) through ChromaDB.
-   Integrate LangChain for using [RAG chain](https://python.langchain.com/v0.1/docs/use_cases/question_answering/) to enhance responses by retrieving relevant intents and knowledge from external data sources.
-   Create custom templates in  `LLaMA Factory/src/llamafactory/data/template.py`  within LLaMA Factory.
-   Use hyperparameter tuning with [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) or other methods for optimization.
