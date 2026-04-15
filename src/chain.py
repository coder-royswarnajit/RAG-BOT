import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt, max_tokens=1024):
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "system", "content": "You are a strict RAG assistant. Use ONLY provided evidence. Do not hallucinate."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def select_and_plan(query, evidence):
    """Use Groq to generate a plan for answering the query based on evidence."""
    evidence_text = "\n".join([f"[Doc: {item.get('source')} | Chunk: {item.get('chunk_id')}]\n{item['text']}"
                                for item in evidence])
    
    prompt = f"""You are a helpful assistant that analyzes documents to answer questions.

            Given the following query and supporting evidence, create a plan for answering the question.
            Identify key points from the evidence that should be included in the answer.

            Query: {query}

            Evidence:
            {evidence_text}

            Please provide:
            1. Key points extracted from the evidence
            2. A brief plan for how to structure the answer
            
            Only use the provided evidence.
            If the answer is not in the evidence, say "I don't know."
            Do NOT add external knowledge.

            Keep your response concise and focused."""

    message = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    plan = {
        "query": query,
        "key_points": message.choices[0].message.content,
        "selected_evidence": evidence
    }
    
    return plan

def generate_answer(plan):
    query = plan["query"]
    evidence = plan["selected_evidence"]
    
    evidence_text = "\n\n".join([
        f"[Source: {e.get('source')} | Chunk: {e.get('chunk_id')}]\n{e['text']}"
        for e in evidence
    ])

    prompt = f"""You are a strict retrieval-augmented assistant.
                Answer the question using ONLY the provided evidence.
                Question:
                {query}

                Evidence:
                {evidence_text}

                Rules:
                - Do NOT use external knowledge
                - If answer not found, say "I don't know"
                - Cite sources using (source, chunk_id)
                Answer:
                """

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Strict RAG mode. No hallucination."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content