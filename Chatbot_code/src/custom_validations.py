def no_disallowed_content(output: str) -> bool:
    disallowed_phrases = [
        "this is legal advice",
        "medical diagnosis",
        "financial recommendation",
        "investment advice",
        "contact your doctor",
        "consult a lawyer",
    ]
    for phrase in disallowed_phrases:
        if phrase.lower() in output.lower():
            return False
    return True
