import fitz  # PyMuPDF wrapper for simplicity
import os
import base64
from app import app
import re
import cv2
import matplotlib.pyplot as plt
import tiktoken
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image(base64_image):
    """
    Uses OpenAI's GPT-4o model to generate a description of the image.
    """
    try:
        response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
            { "role": "system", "content": "Your job is to extract all the information from the images, includng the text. Extract all the text from the image without changing the order or structure of the information. recheck if all the text has been extracted correctly and return in the same presentation and structure as present in the original image. "},
            { "role": "user",
              "content": [
                {"type": "text", "text": "extract ALL the text from the image in the same structure as present in the image. and then after it summarise everything in brief, do not miss anything "},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                  },
                },
              ],
            }
          ],
          max_tokens=300,
        )
        #print("Chat GPT:")
        #print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in image description: {str(e)}"


def extract_images_and_text_from_pdf(pdf_path, output_folder=None):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    if output_folder is None:
        output_folder = os.path.join(app.static_folder, 'extracted_images')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True)

    # Initialize a variable to store the combined text
    combined_text = ""

    # Loop through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        text = page.get_text()

        # Add the text of the current page to combined_text
        combined_text += f"\n\nPage {page_number + 1}:\n{text}"

        # Get the images from the page
        image_list = page.get_images(full=True)

        # Extract and process each image
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
            image_filepath = os.path.join(output_folder, image_filename)

            # Save the image to the output folder
            with open(image_filepath, "wb") as image_file:
                image_file.write(image_bytes)

            # Encode the image to base64
            base64_image = encode_image(image_filepath)

            # Use GPT-4o to describe the image and extract text
            image_description = describe_image(base64_image)

            # Add the image description and reference to combined_text
            combined_text += f"\n\n[Image: {image_filename}]\n{image_description}"

            print(f"Processed {image_filename} on page {page_number + 1}")
    return combined_text

# Function to extract image references from the text
def extract_image_references(text):
    pattern = r"\[Image:\s*(.*?)\]"
    image_references = re.findall(pattern, text)
    return [f"/static/extracted_images/{img}" for img in image_references]


def answer_query(query):
    text_path = os.path.join(app.config['OUTPUT_TEXT'], "combined_text.txt")
    
    # Ensure UTF-8 encoding to avoid UnicodeDecodeError
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators=["\n\n", "\n"]
    )
    
    splits = text_splitter.split_text(text)  # Use split_text instead of loaders.load()
    
    embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_texts(splits, embedding)  # Use from_texts instead of from_documents
    
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name='gpt-4o-mini', temperature=0)

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer and don't find it in the given context, just say that you don't know, 
    don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    result = qa_chain({"query": query})
    
    answer_text = result["result"]  # Extract answer
    
    ret_text = "".join([doc.page_content for doc in result["source_documents"]])
    image_references = extract_image_references(ret_text)
    image_references_list = list(set(image_references))  # Remove duplicates
    
    return {
        "answer": answer_text,
        "images": image_references_list
    }
