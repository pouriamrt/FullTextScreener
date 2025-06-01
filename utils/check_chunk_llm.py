from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from joblib import Memory

memory = Memory("cache_dir/llm", verbose=0)

@memory.cache
def send_to_llm(text, label, description, model="gpt-4.1-mini"):
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a domain expert in clinical research. Your task is to verify whether a given chunk of text from a research paper satisfies a specific inclusion criterion.\n\n"
            "You will be given:\n"
            "- A CHUNK from the paper\n"
            "- A CRITERION LABEL (e.g., 'Study Design')\n"
            "- A CRITERION DESCRIPTION (i.e., what the label actually means)\n\n"
            "Evaluate whether the chunk supports or satisfies the criterion. Respond with YES or NO, and then explain your reasoning in 1–2 sentences."
        ),
        (
            "human",
            "CHUNK:\n{text}\n\n"
            "CRITERION LABEL:\n{label}\n\n"
            "CRITERION DESCRIPTION:\n{description}\n\n"
            "Does the chunk satisfy the criterion? Answer YES or NO and explain."
        ),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "text": text,
        "label": label,
        "description": description
    })

    return response.content
