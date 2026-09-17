import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def evaluate_rag_response(question: str, answer: str, contexts: list[str]):
    """
    Evaluates a RAG response using a custom LLM-as-a-Judge approach.
    Returns a dictionary of scores (0.0 to 1.0) for Faithfulness and Relevance.
    """
    
    # Initialize the judge model
    judge_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    context_text = "\n\n".join(contexts)
    
    # Define the evaluation prompt
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an impartial AI judge evaluating the quality of an answer provided by a RAG system.\n"
         "You must evaluate the answer based on two metrics:\n"
         "1. Faithfulness: Is the answer factually derived strictly from the provided context? (Score 0.0 to 1.0)\n"
         "2. Relevance: Does the answer directly address the user's question? (Score 0.0 to 1.0)\n\n"
         "Respond ONLY with a valid JSON object in this exact format: {{\"faithfulness\": 0.9, \"relevance\": 0.8}}"
        ),
        ("human", 
         "QUESTION: {question}\n\n"
         "RETRIEVED CONTEXT: {context}\n\n"
         "AI ANSWER: {answer}\n\n"
         "Provide your evaluation scores as JSON."
        )
    ])
    
    # Create the chain
    chain = eval_prompt | judge_llm
    
    print("Running custom LLM evaluation...")
    try:
        response = chain.invoke({
            "question": question,
            "context": context_text,
            "answer": answer
        })
        
        # Clean up the response to parse JSON (handling potential markdown blocks)
        raw_output = response.content.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()
            
        scores = json.loads(raw_output)
        return scores
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0}

if __name__ == "__main__":
    print("Testing evaluator...")
    res = evaluate_rag_response(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["France is a country in Europe. Its capital is Paris."]
    )
    print("Evaluation Results:", res)
