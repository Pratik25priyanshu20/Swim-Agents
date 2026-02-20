# tests/test_gee_access.py

"""Smoke test for Google Generative AI connectivity."""

import os
import pytest


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_connectivity():
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    response = model.invoke("Hello, who are you?")
    assert response.content
    assert len(response.content) > 0
