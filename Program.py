# Text Extraction
def Extraction(PDF):
    import fitz

    document = fitz.open(PDF)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return text

# Text Cleaning
def PreProssesText(text):
    Clean_Text = ""

    for i in text:
        if i.isalpha() == True:
            Clean_Text += i + " "
        else:
            pass

    return Clean_Text

# Chunking
def Chunking(Text, Chunk):
    Rslt = []
    chk = ""
    upper = ""
    for i in Text:
        if len(chk) != Chunk and i.isalpha() == True:
            chk += i
        else:
            Rslt.append(chk)
            chk = ""
    
    return Rslt

#Vectorization
def vectorize_text(text_chunks):
    from transformers import TFAutoModel, AutoTokenizer
    import tensorflow as tf
    import numpy as np

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TFAutoModel.from_pretrained("bert-base-uncased")

    encodings = tokenizer(text_chunks, padding=True, truncation=True, return_tensors="tf")
    model_output = model(encodings)
    vectors = model_output.last_hidden_state[:, 0, :].numpy()  # CLS token embedding
    return vectors


PDF = input("Enter PDF Path: ")

Extracted_Text = Extraction(PDF)
Prossesed_Text = PreProssesText(Extracted_Text)
Chunked_Text = Chunking(Prossesed_Text, 1)
Vector_Text = vectorize_text(Chunked_Text)

print("\n")
print("\n Output")
print("\n")
print(Vector_Text)
