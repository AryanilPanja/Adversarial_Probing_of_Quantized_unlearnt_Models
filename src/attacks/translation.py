from deep_translator import GoogleTranslator

def generate_attack(query, target_lang='es'):
    """
    Translates the query to a target language.
    """
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(query)
        return f"Answer the following question: {translated}"
    except Exception as e:
        print(f"Translation failed: {e}")
        return query
