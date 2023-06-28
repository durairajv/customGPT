from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext,StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import os



os.environ["OPENAI_API_KEY"] = '<your Openai Api key>'




def construct_index(directory_path):
   
   docs = SimpleDirectoryReader(directory_path).load_data()

   index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

   index.storage_context.persist(persist_dir="vectorstore")

   return index



def chatbot(input_text):

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="vectorstore")

     #  loading the index from disk
    index = load_index_from_storage(storage_context,service_context= service_context)  

    query_engine = index.as_query_engine(verbose=True)

    response = query_engine.query(input_text)
    
    return response.response



iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Your Shopping Assistant")

iface.launch(share=True)


llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


index = construct_index("docs")

iface.launch(share=True)

