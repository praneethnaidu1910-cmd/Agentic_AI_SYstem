"""
LLM Factory - Supports multiple free and paid providers
Free options: Google Gemini, Hugging Face, Ollama
"""
import os
from typing import Optional
from langchain_core.language_models import BaseChatModel


def create_llm(provider: str = "gemini", model_name: Optional[str] = None, temperature: float = 0.7) -> BaseChatModel:
    """
    Create an LLM instance based on provider
    
    Args:
        provider: "gemini" (FREE), "huggingface" (FREE), "ollama" (FREE), or "openai" (PAID)
        model_name: Optional specific model name
        temperature: LLM temperature
        
    Returns:
        BaseChatModel instance
    """
    provider = provider.lower()
    
    if provider == "gemini":
        return _create_gemini_llm(model_name, temperature)
    elif provider == "huggingface":
        return _create_huggingface_llm(model_name, temperature)
    elif provider == "ollama":
        return _create_ollama_llm(model_name, temperature)
    elif provider == "openai":
        return _create_openai_llm(model_name, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'huggingface', 'ollama', or 'openai'")


def _create_gemini_llm(model_name: Optional[str], temperature: float) -> BaseChatModel:
    """Create Google Gemini LLM (FREE TIER - 60 requests/minute)"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Get a FREE API key from: "
                "https://makersuite.google.com/app/apikey or "
                "https://aistudio.google.com/app/apikey"
            )
        
        # Check for model in environment variable if not provided
        if not model_name:
            model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        model = model_name
        print(f"[OK] Using Google Gemini ({model}) - FREE TIER")
        print("  Free tier: 60 requests/minute, 1,500 requests/day")
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )
    except ImportError:
        raise ImportError(
            "langchain-google-genai not installed. "
            "Install with: pip install langchain-google-genai"
        )
    except Exception as e:
        raise ValueError(f"Error creating Gemini LLM: {e}")


def _create_huggingface_llm(model_name: Optional[str], temperature: float) -> BaseChatModel:
    """Create Hugging Face LLM (FREE with API token)"""
    try:
        from langchain_community.llms import HuggingFaceEndpoint
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not found. Get a FREE token from: "
                "https://huggingface.co/settings/tokens"
            )
        
        model = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"[OK] Using Hugging Face ({model}) - FREE with token")
        
        return HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model}",
            huggingface_api_key=api_key,
            task="text-generation",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 512
            }
        )
    except ImportError:
        raise ImportError(
            "langchain-community not installed. "
            "Install with: pip install langchain-community"
        )
    except Exception as e:
        raise ValueError(f"Error creating Hugging Face LLM: {e}")


def _create_ollama_llm(model_name: Optional[str], temperature: float) -> BaseChatModel:
    """Create Ollama LLM (COMPLETELY FREE - runs locally)"""
    try:
        # Try ChatOllama first (better for chat interactions)
        try:
            from langchain_community.chat_models import ChatOllama
            
            model = model_name or "llama3.2"
            print(f"[OK] Using Ollama ({model}) - COMPLETELY FREE (local)")
            print("  Make sure Ollama is running: ollama serve")
            print(f"  Make sure model is pulled: ollama pull {model}")
            print("  Recommended models for function calling: llama3.2, llama3.1, mistral, mixtral")
            
            return ChatOllama(
                model=model,
                temperature=temperature
            )
        except ImportError:
            # Fallback to regular Ollama LLM
            from langchain_community.llms import Ollama
            
            model = model_name or "llama3.2"
            print(f"[OK] Using Ollama ({model}) - COMPLETELY FREE (local)")
            print("  Make sure Ollama is running: ollama serve")
            print(f"  Make sure model is pulled: ollama pull {model}")
            print("  Recommended models for function calling: llama3.2, llama3.1, mistral, mixtral")
            
            return Ollama(
                model=model,
                temperature=temperature
            )
    except ImportError:
        raise ImportError(
            "langchain-community not installed. "
            "Install with: pip install langchain-community"
        )
    except Exception as e:
        raise ValueError(f"Error creating Ollama LLM: {e}. Make sure Ollama is installed and running.")


def _create_openai_llm(model_name: Optional[str], temperature: float) -> BaseChatModel:
    """Create OpenAI LLM (PAID)"""
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Get an API key from: "
                "https://platform.openai.com/api-keys"
            )
        
        model = model_name or "gpt-4o-mini"
        print(f"[OK] Using OpenAI ({model}) - PAID")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install langchain-openai"
        )
    except Exception as e:
        raise ValueError(f"Error creating OpenAI LLM: {e}")

