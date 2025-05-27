import json
import openai
import time
import random
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DialogDatasetBuilder:
    def __init__(self, api_key: str, base_url: str, model_name: str = "deepseek-ai/DeepSeek-V3"):
        """
        Initializes the dialog dataset builder.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API endpoint.
            model_name: Name of the model to be used, such as "deepseek-ai/DeepSeek-V3".
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.processed_dialogs = []
        
    def load_dailydialog(self, file_path: str) -> List[Dict]:
        """
        Loads the DailyDialog dataset.

        Args:
            file_path: Path to the DailyDialog data file.

        Returns:
            A list of dialogs.
        """
        dialogs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # DailyDialog format: turns are separated by '__eou__'
                        utterances = line.split('__eou__')
                        utterances = [utt.strip() for utt in utterances if utt.strip()]
                        if len(utterances) >= 2:  # At least 2 turns required
                            dialogs.append({
                                'dialog_id': len(dialogs),
                                'utterances': utterances
                            })
            logger.info(f"Successfully loaded {len(dialogs)} dialogs")
            return dialogs
        except Exception as e:
            logger.error(f"Failed to load DailyDialog data: {e}")
            return []
    
    def filter_dialogs(self, dialogs: List[Dict], 
                      min_turns: int = 4, 
                      max_turns: int = 10,
                      sample_size: int = None) -> List[Dict]:
        """
        Filters dialogs based on specified criteria.

        Args:
            dialogs: List of original dialogs.
            min_turns: Minimum number of turns in a dialog.
            max_turns: Maximum number of turns in a dialog.
            sample_size: Number of dialogs to sample. If None, no sampling is performed.

        Returns:
            Filtered list of dialogs.
        """
        filtered = []
        for dialog in dialogs:
            turn_count = len(dialog['utterances'])
            if min_turns <= turn_count <= max_turns:
                filtered.append(dialog)
        
        if sample_size and len(filtered) > sample_size:
            filtered = random.sample(filtered, sample_size)
            
        logger.info(f"After filtering, {len(filtered)} dialogs remain")
        return filtered
    
    def generate_assistant_response(self, context: List[str]) -> str:
        """
        Generates an assistant response using DeepSeek-V3.

        Args:
            context: Context of the conversation.

        Returns:
            Generated response.
        """
        try:
            # Build conversation history
            messages = [
                {"role": "system", "content": "You are a friendly, natural conversational assistant. Generate appropriate responses based on the conversation context, maintaining coherence and naturalness. Responses should be concise and align with everyday conversation habits."}
            ]
            
            # Add conversation history (alternating user and assistant roles)
            for i, utterance in enumerate(context):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": utterance})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Sorry, I can't respond right now."
    
    def process_dialog(self, dialog: Dict) -> tuple:
        """
        Processes a single dialog to generate H-H and H-C versions.

        Args:
            dialog: Original dialog.

        Returns:
            Tuple containing (h_h_dialog, h_c_dialog).
        """
        utterances = dialog['utterances']
        dialog_id = dialog['dialog_id']
        
        # H-H dialog: Keep it unchanged, label as 0
        h_h_dialog = {
            'dialog_id': f"hh_{dialog_id}",
            'utterances': utterances.copy(),
            'label': 0,
            'type': 'human-human'
        }
        
        # H-C dialog: Replace human assistant's replies, label as 1
        h_c_utterances = []
        
        for i, utterance in enumerate(utterances):
            if i % 2 == 0:  # User turn, keep unchanged
                h_c_utterances.append(utterance)
            else:  # Assistant turn, replace with AI-generated response
                context = h_c_utterances.copy()  # Current conversation context
                ai_response = self.generate_assistant_response(context)
                h_c_utterances.append(ai_response)
                
                # Delay between API calls to avoid rate limiting
                time.sleep(0.5)
        
        h_c_dialog = {
            'dialog_id': f"hc_{dialog_id}",
            'utterances': h_c_utterances,
            'label': 1,
            'type': 'human-chatbot'
        }
        
        return h_h_dialog, h_c_dialog
    
    def build_dataset(self, dailydialog_path: str, 
                     output_path: str,
                     min_turns: int = 4,
                     max_turns: int = 10,
                     sample_size: int = 100) -> None:
        """
        Builds a complete dialog dataset.

        Args:
            dailydialog_path: Path to the DailyDialog data file.
            output_path: Path to save the output file.
            min_turns: Minimum number of turns in a dialog.
            max_turns: Maximum number of turns in a dialog.
            sample_size: Number of dialogs to process.
        """
        logger.info("Starting to build the dialog dataset...")
        
        # 1. Load and filter DailyDialog data
        dialogs = self.load_dailydialog(dailydialog_path)
        if not dialogs:
            logger.error("No valid dialog data was loaded")
            return
        
        filtered_dialogs = self.filter_dialogs(
            dialogs, min_turns, max_turns, sample_size
        )
        
        # 2. Process each dialog
        dataset = []
        total = len(filtered_dialogs)
        
        for idx, dialog in enumerate(filtered_dialogs):
            try:
                logger.info(f"Processing dialog {idx+1}/{total}: {dialog['dialog_id']}")
                
                h_h_dialog, h_c_dialog = self.process_dialog(dialog)
                dataset.extend([h_h_dialog, h_c_dialog])
                
                # Save progress every 10 dialogs
                if (idx + 1) % 10 == 0:
                    self._save_progress(dataset, output_path)
                    
            except Exception as e:
                logger.error(f"Error processing dialog {dialog['dialog_id']}: {e}")
                continue
        
        # 3. Save the final dataset
        self._save_dataset(dataset, output_path)
        logger.info(f"Dataset construction completed! Total {len(dataset)} dialog records generated")
        self._print_dataset_stats(dataset)
    
    def _save_progress(self, dataset: List[Dict], output_path: str) -> None:
        """Saves progress."""
        progress_path = f"{output_path}.progress"
        with open(progress_path, 'w', encoding='utf-8') as f:
            for dialog in dataset:
                f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
    
    def _save_dataset(self, dataset: List[Dict], output_path: str) -> None:
        """Saves the final dataset."""
        # Save as JSONL format
        with open(output_path, 'w', encoding='utf-8') as f:
            for dialog in dataset:
                f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
        
        # Also save as JSON format
        json_path = output_path.replace('.jsonl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    def _print_dataset_stats(self, dataset: List[Dict]) -> None:
        """Prints statistics about the dataset."""
        h_h_count = sum(1 for d in dataset if d['label'] == 0)
        h_c_count = sum(1 for d in dataset if d['label'] == 1)
        
        logger.info("="*50)
        logger.info("Dataset Statistics:")
        logger.info(f"Total dialogs: {len(dataset)}")
        logger.info(f"H-H dialogs: {h_h_count}")
        logger.info(f"H-C dialogs: {h_c_count}")
        logger.info(f"Average turns per dialog: {sum(len(d['utterances']) for d in dataset) / len(dataset):.1f}")
        logger.info("="*50)

# Example usage
def main():
    # Configuration parameters
    API_KEY = "your-api-key-here"  # Replace with your API key
    BASE_URL = "http://111.172.228.182:5200/api/v1/"  # Customize your API base URL
    MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # Model being used
    DAILYDIALOG_PATH = "HH-HC/data/dialogues_text.txt"  # Path to DailyDialog data file
    OUTPUT_PATH = "HH-HC/data/dialog_dataset.jsonl"  # Output file path
    
    # Create dataset builder
    builder = DialogDatasetBuilder(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME
    )
    
    # Build the dataset
    builder.build_dataset(
        dailydialog_path=DAILYDIALOG_PATH,
        output_path=OUTPUT_PATH,
        min_turns=4,      # Minimum 4 turns per dialog
        max_turns=8,      # Maximum 8 turns per dialog
        sample_size=50    # Process 50 dialogs (will generate 100 records)
    )

if __name__ == "__main__":
    main()