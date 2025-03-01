# a5_DPO
# Fine-Tuning GPT-2 with Direct Preference Optimization (DPO)

🚀 A project for training a preference-optimized GPT-2 model using Hugging Face’s `DPOTrainer`, uploading it to the Hugging Face Hub, and deploying a web application.

📌 **Model on Hugging Face**: [EkkaratT/a5_DPO_model](https://huggingface.co/EkkaratT/a5_DPO_model)

---

## 📌 Task 1: Finding a Suitable Dataset (0.5 Points)

### 📍 Selected Dataset
For training the model, I selected the **UltraFeedback-Binarized Preferences Cleaned Dataset** from Hugging Face.

👉 **Dataset Link**: [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

### 📍 Suitable Dataset
✅ **Human Preference Rankings** – Contains human-annotated preference data, making it suitable for **preference-based** training.  
✅ **RLHF (Reinforcement Learning from Human Feedback)** – Supports **DPO (Direct Preference Optimization)** training.  
✅ **Preprocessed & Cleaned** – Already formatted for preference-based training, reducing preprocessing effort.  

### 📍 Preprocessing Steps
To prepare the dataset, I:  
1️⃣ **Loaded the dataset** using `datasets.load_dataset()`.  
2️⃣ **Filtered relevant columns** (`prompt`, `chosen`, `rejected`).  
3️⃣ **Converted it into the correct format** for `DPOTrainer`.  

📌 Dataset Preprocessing Code:
from datasets import load_dataset
```python
# Load dataset
dpo_dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

# Select relevant columns
dpo_dataset = dpo_dataset['train'].select_columns(['prompt', 'chosen', 'rejected'])

# Print example
print(dpo_dataset[0])
```

---

## 📌 Task 2: Training a Model with DPOTrainer (1.5 Points)

### 📍 Model Used
✅ **Base Model**: GPT-2  
✅ **Fine-tuned with**: `DPOTrainer` from Hugging Face’s `trl` library  

### 📍 Hyperparameter Experimentation
For training, I adjusted hyperparameters to fit within my **system limitations**.

| Hyperparameter | Value |
|---------------|-------|
| **Learning Rate** | `1e-3` |
| **Batch Size** | `8` |
| **Gradient Accumulation Steps** | `1` |
| **Max Length** | `128` |
| **Max Prompt Length** | `128` |
| **Max Target Length** | `128` |
| **Label Pad Token ID** | `100` |
| **Max Steps** | `200` |

### 📍 Adjustments Due to System Limitations
- 🔹 Reduced `max_length` **from 512 to 128** → **Memory Optimization**  
- 🔹 Reduced `max_steps` **from 1000 to 200** → **Faster Training**  

### 📍 Training Performance
📌 **Final Training Output:**  
```python
TrainOutput(
    global_step=200,
    training_loss=1.6350819182395935,
    metrics={
        'train_runtime': 74.531,
        'train_samples_per_second': 21.468,
        'train_steps_per_second': 2.683,
        'train_loss': 1.6350819182395935,
        'epoch': 1.6
    }
)
```

- ✅ **Final Training Loss**: `1.635`  
- ✅ **Training Time**: `74.5 seconds`  
- ✅ **Training Speed**: `21.46 samples/sec`  

---

## 📌 Task 3: Pushing the Model to Hugging Face Hub (0.5 Points)

### 📍 Saving & Uploading the Model

1️⃣ **Save the fine-tuned model locally**  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Save the model and tokenizer
model.save_pretrained("./test")
tokenizer.save_pretrained("./test")
