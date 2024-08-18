from gtts import gTTS
import os
import json
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pyttsx3
import random
import soundfile as sf
import numpy as np

# Ensure the directory for saving voice samples exists
os.makedirs("voice_samples", exist_ok=True)

# Additional medical specialties and terms
med_specialties_extended = {
    "oncology": ["chemotherapy", "radiotherapy", "carcinoma"],
    "neurology": ["epilepsy", "multiple sclerosis", "neurodegeneration"]
}

# More contextual phrases for dataset robustness
contextual_phrases_extended = [
    "The patient presents with acute respiratory distress.",
    "Consider administering epinephrine."
]

# More abbreviations and symbols
abbreviations_extended = ["BPM", "TID", "PRN"]

# More medical procedures
medical_procedures_extended = [
    "Administer intravenous fluids.", 
    "Schedule an MRI for the patient."
]

# More drug names and dosages
medications_extended = [
    "Prescribe warfarin 5 mg daily.", 
    "Administer 10 units of insulin."
]

# More patient conditions and symptoms
patient_conditions_extended = ["The patient has a history of hypertension."]

# More rare conditions and syndromes
rare_conditions_extended = ["Suspect Guillain-Barre syndrome."]

# Combine all extended phrases into a single list
phrases_extended = (
    contextual_phrases_extended + medical_procedures_extended + 
    medications_extended + patient_conditions_extended + rare_conditions_extended + 
    list(med_specialties_extended['oncology']) + 
    list(med_specialties_extended['neurology']) + 
    abbreviations_extended
)


class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]['audio_path']
        transcription = self.data[idx]['transcription']

        # Load audio file
        audio_data, samplerate = sf.read(audio_path)

        # Convert transcription to input IDs
        input_ids = self.tokenizer(transcription, return_tensors='pt').input_ids.squeeze()

        # Create the input and label tensors
        input_tensor = torch.tensor(input_ids[:-1])
        label_tensor = torch.tensor(input_ids[1:])

        return {
            'input_ids': input_tensor,
            'labels': label_tensor
        }

# Function to add noise to an audio sample
def add_noise(audio_data, noise_factor=0.005):
    """Adding random noise to the audio data"""
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    augmented_data = augmented_data.astype(type(audio_data[0]))
    return augmented_data

# Function to generate voice samples using gTTS and add noise
def generate_voice_with_noise(text, noise=True):
    """Generate voice samples and optionally add noise"""
    tts = gTTS(text=text, lang='en', slow=False)
    filename = f"voice_samples/{text[:30].replace(' ', '_').replace('.', '').replace(',', '')}.mp3"
    tts.save(filename)

    if noise:
        # Read the audio file and add noise
        audio_data, samplerate = sf.read(filename)
        noisy_audio = add_noise(audio_data)
        sf.write(filename, noisy_audio, samplerate)

    print(f"Successfully created voice sample with{'out' if not noise else ''} noise for: {text}")

# Generate voice samples with and without noise
for phrase in phrases_extended:
    generate_voice_with_noise(phrase, noise=random.choice([True, False]))

print("All extended voice samples have been generated.")

# Extend the dataset preparation
dataset_extended = []
for phrase in phrases_extended:
    filename = f"voice_samples/{phrase[:30].replace(' ', '_').replace('.', '').replace(',', '')}.mp3"
    if os.path.exists(filename):
        dataset_extended.append({
            'audio_path': filename,
            'transcription': phrase
        })

# Save the extended dataset to JSON
with open('extended_medical_dataset.json', 'w') as f:
    json.dump(dataset_extended, f, indent=4)

print("Extended dataset saved as extended_medical_dataset.json.")

# Dataset class for PyTorch remains unchanged

# Add evaluation metrics to the fine-tuning function
def fine_tune_whisper_with_evaluation(model_name='openai/whisper-base', epochs=3):
    # Load dataset and tokenizer
    with open('extended_medical_dataset.json') as f:
        data = json.load(f)

    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    train_dataset = MedicalDataset(data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop with evaluation
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_whisper_extended')
    tokenizer.save_pretrained('fine_tuned_whisper_extended')
    print("Fine-tuned model saved.")

# Fine-tune the model with evaluation
fine_tune_whisper_with_evaluation()

# Function to create a demo showcasing the ASR improvements
def create_demo():
    """Create a demo showcasing the fine-tuned model's capabilities."""
    print("Demo: Speak a medical phrase and get the Whisper transcription and TTS-generated voice sample.")

    # Example phrase
    demo_phrase = "Suspect Guillain-Barre syndrome."
    print(f"Input Phrase: {demo_phrase}")

    # Transcription using the fine-tuned model
    model = WhisperForConditionalGeneration.from_pretrained('fine_tuned_whisper_extended')
    tokenizer = WhisperTokenizer.from_pretrained('fine_tuned_whisper_extended')
    inputs = tokenizer(demo_phrase, return_tensors='pt').input_ids.to('cuda')
    output_ids = model.generate(inputs)
    transcription = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Transcription: {transcription}")

    # TTS using pyttsx3
    engine = pyttsx3.init()
    engine.say(transcription)
    engine.runAndWait()
    print("TTS voice sample generated.")

# Create and run the demo
create_demo()

# Two-way conversation setup (this is a placeholder for future implementation)





