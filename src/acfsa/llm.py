from __future__ import annotations


def generate_answer(client, model: str, question: str, context: str, temperature: float = 0.0) -> str:
    """Generate an answer with an OpenAI-compatible chat-completions endpoint."""
    system_prompt = (
        "Ты отвечаешь на вопрос, используя только предоставленный контекст. "
        "Не добавляй факты, которых нет в контексте. Если сведений недостаточно, "
        "прямо скажи об этом. Для каждого существенного утверждения укажи источник "
        "из заголовка соответствующего фрагмента в квадратных скобках."
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()
