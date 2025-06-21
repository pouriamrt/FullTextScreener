from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from joblib import Memory
from utils.cost_tracker import APICostTracker
from config import OPENAI_API_KEY, TEMPERATURE, LLM_SCORE_THRESHOLD

memory = Memory("cache_dir/llm", verbose=0)

@memory.cache
def send_to_llm_batch(chunks, criteria_labels, inclusion_criteria, exclusion_criteria, model='gpt-4.1-mini', max_concurrency=20):
    llm = ChatOpenAI(model=model, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are a domain expert in clinical research. Your task is to assess whether a provided text CHUNK from a research paper satisfies a specified inclusion criterion.\n\n"
            f"You will be given:\n"
            f"- A CHUNK from the paper\n"
            f"- A CRITERION LABEL (e.g., 'Study Design')\n"
            f"- A CRITERION DESCRIPTION (i.e., what the label actually means)\n\n"
            f"Instructions:\n"
            f"1. Evaluate if the chunk provides evidence supporting or satisfying the criterion.\n"
            f"2. Assign a relevance SCORE from 0 to 100.\n"
            f"3. If the SCORE is greater than {LLM_SCORE_THRESHOLD}, consider it RELEVANT.\n"
            f"4. Respond with:\n"
            f"   - YES or NO (is it relevant?)\n"
            f"   - The SCORE (0–100)\n"
            f"   - A brief 1–2 sentence justification."
        ),
        (
            "human",
            "CHUNK:\n{text}\n\n"
            "CRITERION LABEL:\n{label}\n\n"
            "CRITERION DESCRIPTION:\n{description}\n\n"
            "Does the chunk satisfy the criterion?"
        ),
    ])
    
    chain = prompt | llm
    
    batch_inputs = []
    for i, chunk in enumerate(chunks):
        label = chunk['criterion_id']
        description = 'INCLUSION CRITERIA: \n' + inclusion_criteria[label] + "\n\n" + 'EXCLUSION CRITERIA: \n' + exclusion_criteria[label]
        batch_inputs.append({
            "text": chunk['text'],
            "label": criteria_labels[label],
            "description": description
        })
        
    responses = chain.batch(batch_inputs, config={"max_concurrency": max_concurrency})
    
    usages = []
    for r in responses:
        usage = r.response_metadata.get("token_usage", {})
        usages.append(usage)
    
    return responses, usages


@memory.cache
def send_to_llm(text, label, description, model='gpt-4.1-mini'):
    llm = ChatOpenAI(model=model, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)

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
            "Should the chunk be included in the study based on the criterion? Answer YES or NO and explain."
        ),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "text": text,
        "label": label,
        "description": description
    })

    usage = response.response_metadata.get("token_usage", {})

    return response.content, usage
