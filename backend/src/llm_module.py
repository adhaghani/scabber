"""
Language Model Module for response generation
Supports both Ollama and HuggingFace transformers
"""
import requests
import json
from typing import Optional, List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    LLM_MODEL, OLLAMA_MODEL, USE_OLLAMA, SMART_MODEL,
    MAX_NEW_TOKENS, SMART_MODEL_MAX_TOKENS, TEMPERATURE, TOP_P, 
    SYSTEM_PROMPT, SMART_SYSTEM_PROMPT,
    COMPLEXITY_KEYWORDS, THINKING_PHRASES
)


class LanguageModel:
    """Language Model for generating responses using Ollama"""
    
    @staticmethod
    def clean_chunk(text: str) -> str:
        """
        Light cleaning for streaming chunks - preserves spaces
        Only removes emojis and markdown symbols
        """
        import re
        
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove markdown symbols but NOT spaces
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', '', text)
        text = re.sub(r'`+', '', text)
        text = re.sub(r'~+', '', text)
        text = re.sub(r'^#+\s*', '', text)
        
        return text  # Don't strip - preserve spaces!
    
    @staticmethod
    def clean_response(text: str) -> str:
        """
        Clean response text by removing emojis and complex symbols
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text without emojis or special symbols
        """
        import re
        
        # Remove emojis (Unicode emoji ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols extended
            "\U00002600-\U000026FF"  # misc symbols
            "\U00002700-\U000027BF"  # dingbats
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove markdown formatting symbols
        text = re.sub(r'\*{2,}', '', text)  # Remove ** or ***
        text = re.sub(r'_{2,}', '', text)   # Remove __ or ___
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove markdown headers
        text = re.sub(r'`{1,3}', '', text)  # Remove code backticks
        text = re.sub(r'~{2,}', '', text)   # Remove strikethrough ~~
        
        # Remove common special symbols that TTS struggles with
        text = re.sub(r'[•→←↑↓►◄▶◀★☆♦♠♣♥]', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        use_ollama: bool = USE_OLLAMA
    ):
        self.model_name = model_name
        self.base_model_name = model_name  # Store the base (fast) model name
        self.smart_model_name = SMART_MODEL  # Smart model for complex questions
        self.use_ollama = use_ollama
        self.ollama_url = "http://localhost:11434/api/chat"
        self.conversation_history: List[Dict[str, str]] = []
        self.is_loaded = False
        self.smart_model_loaded = False  # Track if smart model is currently active
        
    def is_complex_question(self, user_input: str) -> bool:
        """
        Ask the base AI to evaluate if a question is complex enough for the smart model
        
        Args:
            user_input: The user's question
            
        Returns:
            True if the AI thinks the question warrants deeper thinking
        """
        # Ask the base model to evaluate complexity
        evaluation_prompt = f"""Evaluate if this question requires deep, detailed analysis or if it's a simple question.

Question: "{user_input}"

Reply with ONLY one word: "COMPLEX" if this question requires detailed explanation, multiple steps, technical knowledge, analysis, or thorough reasoning. Reply "SIMPLE" if it's a basic question, greeting, simple fact, or casual conversation.

Your answer:"""

        try:
            payload = {
                "model": self.base_model_name,
                "messages": [{"role": "user", "content": evaluation_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent evaluation
                    "num_predict": 10  # We only need one word
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('message', {}).get('content', '').strip().upper()
                
                # Check if AI thinks it's complex
                is_complex = "COMPLEX" in answer
                
                if is_complex:
                    print(f"🤔 AI assessed: This question needs deeper thinking")
                
                return is_complex
            
        except Exception as e:
            print(f"⚠️ Complexity check failed: {e}")
        
        # Fallback: check word count only
        word_count = len(user_input.split())
        return word_count > 25
    
    def get_thinking_phrase(self) -> str:
        """Get a random thinking phrase for the small model to say"""
        import random
        return random.choice(THINKING_PHRASES)
    
    def load_smart_model(self) -> bool:
        """
        Load the smart model for complex questions
        
        Returns:
            True if successful
        """
        if self.smart_model_loaded:
            return True
            
        print(f"🧠 Loading smart model: {self.smart_model_name}")
        
        try:
            # Check if smart model is available
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check if smart model exists
                if not any(self.smart_model_name in m for m in model_names):
                    print(f"   Smart model not found. Pulling {self.smart_model_name}...")
                    # Trigger model pull
                    pull_response = requests.post(
                        "http://localhost:11434/api/pull",
                        json={"name": self.smart_model_name},
                        timeout=300  # 5 minute timeout for download
                    )
                    if pull_response.status_code != 200:
                        print(f"❌ Failed to pull smart model")
                        return False
                
                # Switch to smart model
                self.model_name = self.smart_model_name
                self.smart_model_loaded = True
                print(f"✅ Smart model ready: {self.smart_model_name}")
                return True
                
        except Exception as e:
            print(f"❌ Error loading smart model: {e}")
            return False
            
        return False
    
    def unload_smart_model(self) -> None:
        """Switch back to the fast base model"""
        if not self.smart_model_loaded:
            return
            
        print(f"🔄 Switching back to fast model: {self.base_model_name}")
        self.model_name = self.base_model_name
        self.smart_model_loaded = False
        print(f"✅ Using fast model: {self.base_model_name}")
    
    def generate_with_model(
        self,
        user_input: str,
        model_name: str,
        temperature: float = TEMPERATURE,
        use_history: bool = True
    ) -> str:
        """
        Generate response using a specific model
        
        Args:
            user_input: The user's message
            model_name: Specific model to use
            temperature: Sampling temperature
            use_history: Whether to include conversation history
            
        Returns:
            Generated response text
        """
        # Build messages
        messages = []
        
        # Use appropriate system prompt based on model
        is_smart = model_name == self.smart_model_name
        system_prompt = SMART_SYSTEM_PROMPT if is_smart else SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        if use_history:
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_input})
        
        # Use higher token limit for smart model
        num_predict = SMART_MODEL_MAX_TOKENS if is_smart else MAX_NEW_TOKENS
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict  # Max tokens to generate
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=180  # 3 minute timeout for smart model
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get('message', {}).get('content', '').strip()
                assistant_message = self.clean_response(assistant_message)
                
                if use_history and assistant_message:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            else:
                return "I'm sorry, I encountered an error generating a response."
                
        except requests.exceptions.Timeout:
            return "I'm sorry, the response took too long. Please try again."
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm sorry, I encountered an error."
        
    def load(self) -> None:
        """Verify Ollama is running and model is available"""
        print(f"🧠 Loading Language Model: {self.model_name}")
        
        if self.use_ollama:
            print("   Using Ollama backend")
            
            # Check if Ollama is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'].split(':')[0] for m in models]
                    
                    # Check if our model is available
                    if self.model_name.split(':')[0] in model_names or any(self.model_name in m['name'] for m in models):
                        print(f"✅ Model '{self.model_name}' is available")
                        self.is_loaded = True
                    else:
                        print(f"⚠️  Model '{self.model_name}' not found. Available models: {model_names}")
                        print(f"   Run: ollama pull {self.model_name}")
                        # Try to use the model anyway, Ollama might auto-pull
                        self.is_loaded = True
                else:
                    raise ConnectionError("Ollama API returned error")
                    
            except requests.exceptions.ConnectionError:
                print("❌ Ollama is not running!")
                print("   Start Ollama with: ollama serve")
                raise RuntimeError("Ollama server not running. Start with 'ollama serve'")
        else:
            # Load HuggingFace model (fallback)
            self._load_transformers_model()
            
        print("✅ Language Model ready")
    
    def _load_transformers_model(self):
        """Load model using HuggingFace transformers"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from config import DEVICE, LOW_CPU_MEM_USAGE
        
        self.device = DEVICE
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.is_loaded = True
        
    def generate_response(
        self,
        user_input: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        use_history: bool = True
    ) -> str:
        """
        Generate a response to user input
        
        Args:
            user_input: The user's message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_history: Whether to include conversation history
            
        Returns:
            Generated response text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.use_ollama:
            return self._generate_ollama(user_input, temperature, use_history)
        else:
            return self._generate_transformers(user_input, max_new_tokens, temperature, top_p, use_history)
    
    def _generate_ollama(
        self,
        user_input: str,
        temperature: float,
        use_history: bool
    ) -> str:
        """Generate response using Ollama API"""
        
        # Build messages
        messages = []
        
        # Use appropriate system prompt based on model
        system_prompt = SMART_SYSTEM_PROMPT if self.smart_model_loaded else SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if use_history:
            messages.extend(self.conversation_history[-6:])
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use higher token limit for smart model
        num_predict = SMART_MODEL_MAX_TOKENS if self.smart_model_loaded else MAX_NEW_TOKENS
        
        # Make API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict  # Max tokens to generate
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120  # 2 minute timeout for response
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get('message', {}).get('content', '').strip()
                
                # Clean response (remove emojis and complex symbols)
                assistant_message = self.clean_response(assistant_message)
                
                # Update conversation history
                if use_history and assistant_message:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return "I'm sorry, I encountered an error generating a response."
                
        except requests.exceptions.Timeout:
            return "I'm sorry, the response took too long. Please try again."
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm sorry, I encountered an error."
    
    def generate_response_stream(
        self,
        user_input: str,
        temperature: float = TEMPERATURE,
        use_history: bool = True
    ):
        """
        Generate a streaming response to user input
        
        Args:
            user_input: The user's message
            temperature: Sampling temperature
            use_history: Whether to include conversation history
            
        Yields:
            Chunks of generated response text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not self.use_ollama:
            # Fallback to non-streaming for transformers
            yield self._generate_transformers(user_input, MAX_NEW_TOKENS, temperature, TOP_P, use_history)
            return
        
        # Build messages
        messages = []
        
        # Use appropriate system prompt based on model
        system_prompt = SMART_SYSTEM_PROMPT if self.smart_model_loaded else SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        if use_history:
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_input})
        
        # Use higher token limit for smart model
        num_predict = SMART_MODEL_MAX_TOKENS if self.smart_model_loaded else MAX_NEW_TOKENS
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": temperature,
                "num_predict": num_predict  # Max tokens to generate
            }
        }
        
        full_response = ""
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120,
                stream=True  # Stream the HTTP response
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                chunk = data['message']['content']
                                # Light cleaning - preserve spaces between words
                                chunk = self.clean_chunk(chunk)
                                if chunk:  # Only yield non-empty chunks
                                    full_response += chunk
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
                
                # Update conversation history after complete
                if use_history and full_response:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": full_response})
            else:
                yield "I'm sorry, I encountered an error generating a response."
                
        except requests.exceptions.Timeout:
            yield "I'm sorry, the response took too long. Please try again."
        except Exception as e:
            print(f"❌ Error: {e}")
            yield "I'm sorry, I encountered an error."
    
    def _generate_transformers(
        self,
        user_input: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_history: bool
    ) -> str:
        """Generate response using HuggingFace transformers"""
        import torch
        
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        if use_history:
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = self._format_prompt_fallback(messages)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if use_history:
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _format_prompt_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback prompt formatting"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")
    
    def is_using_smart_model(self) -> bool:
        """Check if currently using the smart model"""
        return self.smart_model_loaded
        
    def unload(self) -> None:
        """Unload the model to free memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        print("🗑️  Language Model unloaded")


def test_llm():
    """Test LLM module"""
    print("\n🧠 Language Model Test")
    print("=" * 40)
    
    llm = LanguageModel()
    llm.load()
    
    # Test conversation
    test_inputs = [
        "Hello! How are you today?",
        "Tell me a short joke."
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        response = llm.generate_response(user_input)
        print(f"🤖 Assistant: {response}")
    
    llm.unload()
    print("\n✅ LLM test complete!")


if __name__ == "__main__":
    test_llm()
