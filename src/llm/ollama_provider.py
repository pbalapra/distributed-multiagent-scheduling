#!/usr/bin/env python3
"""
Ollama Local LLM Provider
========================

Provider for local Ollama models (like Mistral) running on localhost.
Supports both synchronous and asynchronous calls to local Ollama API.
"""

import json
import requests
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .llm_interface import LLMInterface, LLMRequest, LLMResponse, LLMProvider


class OllamaProvider(LLMInterface):
    """Provider for local Ollama models"""
    
    def __init__(self, 
                 model_name: str = "mistral",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 90.0,
                 max_retries: int = 3):
        """
        Initialize Ollama provider
        
        Args:
            model_name: Name of the Ollama model (e.g., 'mistral', 'llama2', 'codellama')
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        print(f"🦙 Ollama Provider initialized (model: {model_name}, url: {base_url})")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                # Check if our model is available
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                # Check for exact match or partial match (for versioned models)
                model_available = any(
                    self.model_name in model_name 
                    for model_name in available_models
                )
                
                if model_available:
                    print(f"✅ Ollama model '{self.model_name}' is available")
                    return True
                else:
                    print(f"⚠️ Ollama model '{self.model_name}' not found. Available: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            return False
    
    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Generate response synchronously"""
        for attempt in range(self.max_retries):
            try:
                # Prepare request payload
                payload = {
                    "model": self.model_name,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": getattr(request, 'temperature', 0.1),
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }
                
                # Add max_tokens if specified
                if hasattr(request, 'max_tokens') and request.max_tokens:
                    payload["options"]["num_predict"] = request.max_tokens
                
                # Make request to Ollama
                print(f"\n🤖 [LLM REQUEST] Task: {request.task_type}")
                print(f"📝 FULL PROMPT:\n{'-'*50}")
                print(payload['prompt'])
                print(f"{'-'*50}")
                print(f"⚙️ Config: temp={payload['options']['temperature']}, tokens={payload['options'].get('num_predict', 'unlimited')}")
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=getattr(request, 'timeout', self.timeout)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response text
                    content = result.get("response", "").strip()
                    
                    # Log the response
                    response_time = result.get("total_duration", 0) / 1e9
                    tokens_used = result.get("eval_count", 0)
                    print(f"✅ [LLM RESPONSE] Time: {response_time:.1f}s, Tokens: {tokens_used}")
                    print(f"📄 FULL RESPONSE:\n{'-'*50}")
                    print(content)
                    print(f"{'-'*50}")
                    
                    # Try to extract JSON if the response looks like JSON
                    if content.startswith('{') and content.endswith('}'):
                        try:
                            # Validate JSON
                            json.loads(content)
                        except json.JSONDecodeError:
                            # If JSON is malformed, wrap it
                            content = self._fix_json_response(content, request.task_type)
                    elif request.task_type in ["job_scoring", "fault_recovery", "negotiation"]:
                        # For structured tasks, ensure we return valid JSON
                        content = self._create_fallback_json(request.task_type, content)
                    
                    return LLMResponse(
                        content=content,
                        reasoning=self._extract_reasoning(content),
                        metadata={
                            "model": self.model_name,
                            "provider": "ollama",
                            "tokens_used": result.get("eval_count", 0),
                            "response_time": result.get("total_duration", 0) / 1e9,  # Convert to seconds
                            "attempt": attempt + 1
                        }
                    )
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ Ollama request failed after {self.max_retries} attempts: {e}")
                    # Return fallback response
                    return self._create_fallback_response(request)
                else:
                    print(f"⚠️ Ollama attempt {attempt + 1} failed, retrying: {e}")
                    continue
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response (async version)"""
        return await self.generate_async(request)
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Generate response asynchronously"""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": getattr(request, 'temperature', 0.1),
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }
                
                if hasattr(request, 'max_tokens') and request.max_tokens:
                    payload["options"]["num_predict"] = request.max_tokens
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=getattr(request, 'timeout', self.timeout))
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            content = result.get("response", "").strip()
                            
                            # Handle JSON responses
                            if content.startswith('{') and content.endswith('}'):
                                try:
                                    json.loads(content)
                                except json.JSONDecodeError:
                                    content = self._fix_json_response(content, request.task_type)
                            elif request.task_type in ["job_scoring", "fault_recovery", "negotiation"]:
                                content = self._create_fallback_json(request.task_type, content)
                            
                            return LLMResponse(
                                content=content,
                                reasoning=self._extract_reasoning(content),
                                metadata={
                                    "model": self.model_name,
                                    "provider": "ollama",
                                    "tokens_used": result.get("eval_count", 0),
                                    "response_time": result.get("total_duration", 0) / 1e9,
                                    "attempt": attempt + 1
                                }
                            )
                        else:
                            raise Exception(f"Ollama API error: {response.status}")
                            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ Async Ollama request failed: {e}")
                    return self._create_fallback_response(request)
                else:
                    print(f"⚠️ Async Ollama attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1)  # Brief delay before retry
    
    def _fix_json_response(self, content: str, task_type: str) -> str:
        """Attempt to fix malformed JSON responses"""
        try:
            # Common JSON fixes
            content = content.replace("'", '"')  # Replace single quotes
            content = content.strip()
            
            # Remove trailing commas
            import re
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            # Validate again
            json.loads(content)
            return content
        except:
            # If still can't parse, create fallback
            return self._create_fallback_json(task_type, content)
    
    def _create_fallback_json(self, task_type: str, original_response: str) -> str:
        """Create fallback structured JSON for different task types"""
        if task_type == "job_scoring":
            return json.dumps({
                "score": 0.5,
                "factors": {"analysis": "fallback"},
                "recommendation": "consider",
                "original_response": original_response[:200]
            })
        elif task_type == "fault_recovery":
            return json.dumps({
                "strategy": "retry_with_alternative",
                "action": "immediate_retry",
                "delay_seconds": 0,
                "escalate": False,
                "original_response": original_response[:200]
            })
        elif task_type == "negotiation":
            return json.dumps({
                "accept": True,
                "counter_offer": {},
                "reasoning_factors": {"analysis": "fallback"},
                "original_response": original_response[:200]
            })
        else:
            return json.dumps({
                "response": original_response,
                "type": task_type
            })
    
    def _create_fallback_response(self, request: LLMRequest) -> LLMResponse:
        """Create fallback response when all attempts fail"""
        fallback_content = self._create_fallback_json(request.task_type, "Service unavailable")
        
        return LLMResponse(
            content=fallback_content,
            reasoning="Fallback response - Ollama service unavailable",
            metadata={
                "model": self.model_name,
                "provider": "ollama_fallback",
                "error": "Service unavailable",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from response if available"""
        try:
            if content.startswith('{'):
                data = json.loads(content)
                # Look for common reasoning fields
                for key in ["reasoning", "rationale", "explanation", "analysis"]:
                    if key in data and data[key]:
                        return str(data[key])
            return None
        except:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(f"{self.base_url}/api/show", 
                                  json={"name": self.model_name}, 
                                  timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ Could not get model info: {e}")
        return {"name": self.model_name, "status": "unknown"}


# Register Ollama provider
def create_ollama_provider(model_name: str = "mistral", **kwargs) -> OllamaProvider:
    """Factory function to create Ollama provider"""
    return OllamaProvider(model_name=model_name, **kwargs)


# Add to available providers
if hasattr(LLMProvider, 'OLLAMA'):
    pass  # Already exists
else:
    # Extend the LLMProvider enum if needed
    LLMProvider.OLLAMA = "ollama"