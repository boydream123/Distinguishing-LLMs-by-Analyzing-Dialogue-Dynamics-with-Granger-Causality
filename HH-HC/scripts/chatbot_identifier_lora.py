import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
import random
import os
import requests


# Suppress warnings from statsmodels and tokenizers
warnings.filterwarnings("ignore")
# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Constants and Configuration ---

# USER: SPECIFY YOUR PREFERRED LOCAL PATH FOR THE MODEL
LOCAL_MODEL_PATH = "./qwen2_7b_instruct_model/" # <--- !!! UPDATE THIS PATH !!!
# Fallback Hugging Face model ID
HF_MODEL_ID = "Qwen/Qwen2-7B-Instruct"

MAX_SEQ_LENGTH = 512
GCT_MAX_LAG = 2
ALPHA_SIG = 0.05
D_GCT = 2
NUM_SEMANTIC_DEFICIENCIES = 4
MIN_TURNS_FOR_GCT = GCT_MAX_LAG + 2

TRAIN_TEST_SPLIT_RATIO = 0.2
BATCH_SIZE = 2 # Adjust based on your GPU memory
EPOCHS = 3
LEARNING_RATE = 2e-5 # LoRA fine-tuning might benefit from a slightly higher LR sometimes, e.g., 1e-4, 2e-4
DATASET_FILE_PATH = 'HH-HC/data/dialog_dataset.jsonl' 

# --- API Configuration (User Provided) ---
# WARNING: It's generally not recommended to hardcode API keys directly in scripts.
# Consider using environment variables or a configuration file for better security.
API_KEY = "your-actual-api-key"  # <--- USER: REPLACE WITH YOUR ACTUAL API KEY
BASE_URL = "xxxx" # <--- USER: VERIFY THIS BASE URL
LLM_MODEL_NAME_FOR_SEMANTIC = "xxxxx" # Model for semantic attribution,such as "deepseek-ai/DeepSeek-V3, Qwen/Qwen2-7B-Instruct"







# --- LoRA Configuration ---
LORA_R = 16  # Rank of the LoRA matrices (common values: 8, 16, 32, 64)
LORA_ALPHA = 32 # Alpha scaling factor (often LORA_R * 2)
# Target modules for Qwen2. May need adjustment based on exact model architecture.
# Common targets are attention projection layers and sometimes MLP layers.
# Inspect model.named_modules() to find exact names if needed.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # "gate_proj", # Uncomment if you want to target MLP layers too
    # "up_proj",   # Uncomment if you want to target MLP layers too
    # "down_proj"  # Uncomment if you want to target MLP layers too
]
LORA_DROPOUT = 0.05 # Dropout for LoRA layers
LORA_BIAS = "none" # Whether to train biases: "none", "all", "lora_only"


# --- NLTK VADER Lexicon Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("VADER lexicon found.")
except LookupError:
    print("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
    
# --- 1. Feature Extraction Helpers ---
def get_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def extract_dialogue_time_series(utterances):
    user_sentiments = []
    agent_sentiments = []
    for i, utt in enumerate(utterances):
        sentiment = get_sentiment_vader(utt)
        if i % 2 == 0: user_sentiments.append(sentiment)
        else: agent_sentiments.append(sentiment)
    return user_sentiments, agent_sentiments

def compute_gct_p_values(user_series, agent_series, max_lag=GCT_MAX_LAG):
    p_values = [1.0, 1.0]
    if len(user_series) < MIN_TURNS_FOR_GCT or len(agent_series) < MIN_TURNS_FOR_GCT: return p_values
    try:
        data_ua = np.array([agent_series, user_series]).T
        if not np.all(np.isclose(data_ua, data_ua[0,:], atol=1e-8), axis=0).any() and np.linalg.matrix_rank(data_ua) == data_ua.shape[1]:
            gct_results_ua = grangercausalitytests(data_ua, [max_lag], verbose=False)
            p_values[0] = gct_results_ua[max_lag][0]['ssr_ftest'][1]
    except Exception: pass
    try:
        data_au = np.array([user_series, agent_series]).T
        if not np.all(np.isclose(data_au, data_au[0,:], atol=1e-8), axis=0).any() and np.linalg.matrix_rank(data_au) == data_au.shape[1]:
            gct_results_au = grangercausalitytests(data_au, [max_lag], verbose=False)
            p_values[1] = gct_results_au[max_lag][0]['ssr_ftest'][1]
    except Exception: pass
    p_values = [1.0 if np.isnan(p) or p is None else p for p in p_values]
    return p_values

def generate_semantic_attributions_placeholder(dialogue_text):
    return [random.randint(0, 1) for _ in range(NUM_SEMANTIC_DEFICIENCIES)]


# --- Updated Semantic Attribution Function ---

def get_semantic_attributions_from_api(dialogue_text, api_key, base_url, model_name):
    """
    Calls an external LLM API to get semantic deficiency attributions.
    Parses the response and returns a binary vector.
    """
    # Define the semantic deficiencies and their corresponding keys
    # (Order matters for the binary vector)
    deficiency_map = {
        "c_goal": 0,    # Goal Obfuscation/Failure
        "c_fact": 1,    # Factual Inconsistency
        "c_common": 2,  # Commonsense Violation
        "c_logic": 3    # Logical Incoherence
    }
    num_deficiencies = len(deficiency_map)
    default_attributions = [0] * num_deficiencies # Default if API fails or no deficiencies

    # Construct the prompt based on the paper
    prompt_text = f"""Input Dialogue:
{dialogue_text}

Contextual Focus (if identifiable as potential H-C): Contributions from the suspected chatbot.

Question: Which of the following pragmatic semantic deficiencies does this dialogue exhibit, particularly concerning the contextual focus if applicable?
1. Goal Obfuscation/Failure (c_goal): The primary user's goals seem unmet, poorly addressed, or significantly side-tracked.
2. Factual Inconsistency (c_fact): The dialogue contains statements that are demonstrably false, misleading, or internally inconsistent with established facts.
3. Commonsense Violation (c_common): The dialogue includes statements, reasoning, or assumptions that clearly contradict basic, everyday commonsense.
4. Logical Incoherence (c_logic): The dialogue displays internal contradictions in reasoning, significant logical fallacies, or a breakdown in coherent argumentation.
If multiple deficiencies are applicable, provide a comma-separated list of the corresponding labels (e.g., "c_goal, c_common").
Answer "None" if none of the options apply.

Your Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Assuming a chat completions endpoint like OpenAI's
    # Adjust the payload if your API expects a different format
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 50, # Adjust as needed, response should be short
        "temperature": 0.2, # Lower temperature for more deterministic output
    }

    api_endpoint = f"{base_url.rstrip('/')}/chat/completions"

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30) # 30-second timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        api_response_json = response.json()
        
        # --- Parse the API response ---
        # This part is highly dependent on the actual structure of your API's response.
        # Assuming it's similar to OpenAI, where the content is in:
        # response_json['choices'][0]['message']['content']
        
        if api_response_json.get("choices") and len(api_response_json["choices"]) > 0:
            content = api_response_json["choices"][0].get("message", {}).get("content", "").strip().lower()
            
            attributions = list(default_attributions) # Start with all zeros
            if content and content != "none":
                detected_labels = [label.strip() for label in content.split(',')]
                for label_key in detected_labels:
                    if label_key in deficiency_map:
                        attributions[deficiency_map[label_key]] = 1
            return attributions
        else:
            print(f"Warning: API response for semantic attribution was empty or malformed. Full response: {api_response_json}")
            return default_attributions

    except requests.exceptions.RequestException as e:
        print(f"Warning: API call for semantic attribution failed. Error: {e}. Returning default attributions.")
        return default_attributions
    except Exception as e:
        print(f"Warning: An unexpected error occurred while processing API response for semantic attribution. Error: {e}. Returning default attributions.")
        return default_attributions

# --- 2. Dataset Class ---
class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder_main):
        self.data = data
        self.tokenizer = tokenizer
        self.le_label_main = label_encoder_main
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        main_label = self.le_label_main.transform([item['label']])[0]
        dialogue_text = " ".join(item['utterances'])
        inputs = self.tokenizer(dialogue_text, max_length=MAX_SEQ_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
        gct_p_values = compute_gct_p_values(*extract_dialogue_time_series(item['utterances']), max_lag=GCT_MAX_LAG)
        return {
            'dialog_id': item['dialog_id'],
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'gct_p_values': torch.tensor(gct_p_values, dtype=torch.float),
            'main_label': torch.tensor(main_label, dtype=torch.long),
            'gct_significance_labels': torch.tensor([1 if p < ALPHA_SIG else 0 for p in gct_p_values], dtype=torch.float),
            'semantic_deficiency_labels': torch.tensor(get_semantic_attributions_from_api(dialogue_text,API_KEY,BASE_URL,LLM_MODEL_NAME_FOR_SEMANTIC), dtype=torch.float)
        }


# --- 3. Model Class (with LoRA) ---
class ChatbotIDModel(nn.Module):
    def __init__(self, local_model_path, hf_model_id, num_main_labels, d_gct_features=D_GCT, num_semantic_deficiencies=NUM_SEMANTIC_DEFICIENCIES):
        super(ChatbotIDModel, self).__init__()

        base_model = None
        model_config_obj = None
        try:
            print(f"Attempting to load base model config from local path: {local_model_path}")
            if not (os.path.exists(local_model_path) and os.path.isdir(local_model_path)):
                raise OSError(f"Local path {local_model_path} does not exist or is not a directory.")
            model_config_obj = AutoConfig.from_pretrained(local_model_path) # Load config first
            print(f"Attempting to load base model from local path: {local_model_path}")
            base_model = AutoModel.from_pretrained(local_model_path, config=model_config_obj)
            print("Base model loaded successfully from local path.")
        except OSError as e:
            print(f"Could not load base model from local path: {local_model_path}. Error: {e}. Falling back to Hugging Face Hub: {hf_model_id}")
            model_config_obj = AutoConfig.from_pretrained(hf_model_id)
            base_model = AutoModel.from_pretrained(hf_model_id, config=model_config_obj)
            print(f"Base model downloaded/loaded from Hugging Face Hub: {hf_model_id}")
            try:
                if not os.path.exists(local_model_path): os.makedirs(local_model_path, exist_ok=True)
                print(f"Attempting to save base model and config to {local_model_path} for future use...")
                base_model.save_pretrained(local_model_path)
                model_config_obj.save_pretrained(local_model_path) # Save config too
                print(f"Base model and config saved to {local_model_path}")
            except Exception as save_e:
                print(f"Could not save base model/config to {local_model_path}. Error: {save_e}.")

        # --- Apply LoRA ---
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            # task_type can be omitted if we are using AutoModel and then adding heads
            # If using AutoModelForSequenceClassification, TaskType.SEQ_CLS would be set
        )
        self.qwen2_peft = get_peft_model(base_model, lora_config)
        print("LoRA applied to the base model.")
        self.qwen2_peft.print_trainable_parameters() # Shows how many parameters are trainable

        self.hidden_size = model_config_obj.hidden_size
        self.fused_feature_size = self.hidden_size + d_gct_features

        # These heads are separate from the LoRA-modified base model and will be trained fully.
        self.class_predictor = nn.Linear(self.fused_feature_size, num_main_labels)
        self.semantic_predictor = nn.Linear(self.hidden_size, num_semantic_deficiencies)
        self.gct_interaction_predictor = nn.Linear(self.fused_feature_size, d_gct_features)

    def forward(self, input_ids, attention_mask, gct_p_values):
        # Use the PEFT model (LoRA adapted base model)
        outputs = self.qwen2_peft(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden_state = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = torch.max(seq_lengths, torch.zeros_like(seq_lengths))
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.size(0), device=input_ids.device), seq_lengths]
        fused_features = torch.cat((pooled_output, gct_p_values), dim=1)

        main_class_logits = self.class_predictor(fused_features)
        semantic_deficiency_logits = self.semantic_predictor(pooled_output)
        gct_significance_logits = self.gct_interaction_predictor(fused_features)
        return main_class_logits, semantic_deficiency_logits, gct_significance_logits
    
    
    # --- 4. Training and Evaluation Functions ---
def train_model_epoch(model, dataloader, optimizer, device, epoch_num, total_epochs):
    model.train() # Set model to training mode (activates LoRA layers for training)
    criterion_class = nn.CrossEntropyLoss()
    criterion_semantic = nn.BCEWithLogitsLoss()
    criterion_gct = nn.BCEWithLogitsLoss()
    total_loss_epoch = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask, gct_p_values, main_labels, semantic_labels, gct_labels = \
            batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['gct_p_values'].to(device), \
            batch['main_label'].to(device), batch['semantic_deficiency_labels'].to(device), batch['gct_significance_labels'].to(device)
        optimizer.zero_grad()
        main_logits, semantic_logits, gct_logits = model(input_ids, attention_mask, gct_p_values)
        loss_class = criterion_class(main_logits, main_labels)
        loss_semantic = criterion_semantic(semantic_logits, semantic_labels)
        loss_gct = criterion_gct(gct_logits, gct_labels)
        total_loss = loss_class + loss_semantic + loss_gct
        total_loss.backward()
        optimizer.step()
        total_loss_epoch += total_loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch_num+1}/{total_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} "
                  f"(Lc: {loss_class.item():.4f}, Ls: {loss_semantic.item():.4f}, Lg: {loss_gct.item():.4f})")
    avg_epoch_loss = total_loss_epoch / len(dataloader)
    print(f"Epoch {epoch_num+1} average training loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def evaluate_model(model, dataloader, device, le_main):
    model.eval() # Set model to evaluation mode (important for dropout and LoRA)
    all_main_preds, all_main_labels, all_main_probs = [], [], []
    criterion_class = nn.CrossEntropyLoss()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, gct_p_values, main_labels = \
                batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['gct_p_values'].to(device), batch['main_label'].to(device)
            main_logits, _, _ = model(input_ids, attention_mask, gct_p_values)
            total_eval_loss += criterion_class(main_logits, main_labels).item()
            preds = torch.argmax(main_logits, dim=1)
            positive_class_index = 1 if 1 in le_main.classes_ else 0 # Default assumption for positive class
            if 1 in le_main.classes_: positive_class_index = list(le_main.classes_).index(1)
            probs = torch.softmax(main_logits, dim=1)[:, positive_class_index]
            all_main_preds.extend(preds.cpu().numpy())
            all_main_labels.extend(main_labels.cpu().numpy())
            all_main_probs.extend(probs.cpu().numpy())
    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = accuracy_score(all_main_labels, all_main_preds)
    f1 = f1_score(all_main_labels, all_main_preds, average='binary' if len(le_main.classes_) == 2 else 'weighted', zero_division=0)
    auroc = float('nan')
    try:
        if len(le_main.classes_) == 2: auroc = roc_auc_score(all_main_labels, all_main_probs)
        else: print("AUROC for multi-class needs OvR or similar.")
    except ValueError as e: print(f"AUROC compute error: {e}")
    print(f"\nTest Set Eval: Loss (L_class): {avg_eval_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}\n")
    return accuracy, f1, auroc, avg_eval_loss


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Local model path: {LOCAL_MODEL_PATH}, Fallback HuggingFace ID: {HF_MODEL_ID}")
    raw_data = []
    try:
        with open(DATASET_FILE_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f): raw_data.append(json.loads(line))
        print(f"Loaded {len(raw_data)} dialogues.")
    except FileNotFoundError: print(f"Error: {DATASET_FILE_PATH} not found."); exit()
    if not raw_data: print("No data. Exiting."); exit()

    all_original_labels = [item['label'] for item in raw_data]
    le_main = LabelEncoder()
    le_main.fit(all_original_labels)
    NUM_MAIN_LABELS_DETECTED = len(le_main.classes_)
    print(f"Detected {NUM_MAIN_LABELS_DETECTED} main classes: {le_main.classes_}")

    try:
        train_data, test_data = train_test_split(raw_data, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42, stratify=all_original_labels)
    except ValueError as e:
        print(f"Stratify warning: {e}. Splitting without stratification.")
        train_data, test_data = train_test_split(raw_data, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    tokenizer = None
    try:
        if not (os.path.exists(LOCAL_MODEL_PATH) and os.path.isdir(LOCAL_MODEL_PATH)): raise OSError("Local path not valid.")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        print("Tokenizer loaded locally.")
    except OSError as e:
        print(f"Local tokenizer load fail: {e}. Downloading from {HF_MODEL_ID}.")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        try:
            if not os.path.exists(LOCAL_MODEL_PATH): os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            print(f"Tokenizer saved to {LOCAL_MODEL_PATH}")
        except Exception as se: print(f"Tokenizer save fail: {se}")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; print("Set pad_token to eos_token.")

    train_dataset = DialogueDataset(train_data, tokenizer, le_main)
    test_dataset = DialogueDataset(test_data, tokenizer, le_main)
    if not train_dataset: print("Train dataset empty. Exit."); exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if test_dataset else None

    model = ChatbotIDModel(LOCAL_MODEL_PATH, HF_MODEL_ID, NUM_MAIN_LABELS_DETECTED).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # Optimizer will only update trainable params (LoRA + heads)

    print("\nStarting training with LoRA...")
    for epoch in range(EPOCHS):
        train_model_epoch(model, train_dataloader, optimizer, device, epoch, EPOCHS)
        if test_dataloader:
            print(f"\nEvaluating on test set after epoch {epoch+1}...")
            evaluate_model(model, test_dataloader, device, le_main)
    print("\nTraining finished.")
    if test_dataloader: print("\nFinal evaluation..."); evaluate_model(model, test_dataloader, device, le_main)
    else: print("\nNo test data for final eval.")