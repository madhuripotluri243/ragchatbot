# Step1: Read a PDF and convert to Text

from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate

from PyPDF2 import PdfReader

text=""

pdf_reader=PdfReader("Gen_ai_basics.pdf")

for page in pdf_reader.pages:
    text=text+page.extract_text() + "\n"

# Step2: Chunk the Text

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

texts = text_splitter.split_text(text)


# Step3: Initializing Vector DB

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone()

index_name = "ragchatbot"

pc.create_index(
    name=index_name,
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

index=pc.Index(index_name)


# Step4: Convert Chunks to Embeddings and Store them in Vector DB

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

for i, chunk in enumerate(texts):
    chunk_embedding=embeddings.embed_query(chunk)
    index.upsert([(str(i), chunk_embedding,{"chunk_text":chunk})])

# Step5: Create LLM Object. Retrieve Top K and Generate Answer

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(temperature=0, model="gpt-4o")

while True:
    query=input("Ask a Question (or type exit to quit): ")
    if query.lower() == "exit":
        break

# Step6: Retrieve Top K
    query_embedding=embeddings.embed_query(query)
    result=index.query(vector=query_embedding,top_k=3, include_metadata=True)

# Step7: Combine All Results in a Single String

    augmented_text="\n\n".join([match.metadata["chunk_text"] for match in result.matches])

# step8: Create Prompt and Call LLM

    prompt_text=""""
    You are an helpful assistant. Answer the question based on the context provided below. If you don't know the answer, say you don't know.

    "Context": {context}
    "User Question": {question}
    """

    prompt=PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text
    )

    # Step9: Chain them Together

    chain=prompt | llm

    response=chain.invoke({"context":augmented_text, "question":query})

    print(f"Retrieved Text: {augmented_text}")
    print(f"Answer is: {response.content}")

