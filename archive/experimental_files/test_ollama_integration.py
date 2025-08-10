#!/usr/bin/env python3
"""
Test Script for Ollama LLM Integration
=====================================

This script tests the Ollama provider integration with various scenarios.
"""

import sys
import os
import json
import asyncio
import time
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from llm.ollama_provider import OllamaProvider
from llm.llm_interface import LLMRequest, LLMResponse, LLMManager, LLMProvider


class OllamaIntegrationTest:
    def __init__(self):
        self.provider = None
        self.test_results = []
        self.manager = LLMManager()
    
    def log_result(self, test_name: str, success: bool, message: str, duration: float = 0):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if duration > 0:
            print(f"   Duration: {duration:.2f}s")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
    
    def test_ollama_availability(self) -> bool:
        """Test if Ollama service is available"""
        print("\n🔍 Testing Ollama Service Availability")
        print("-" * 50)
        
        try:
            # Test with different common models
            models_to_test = ["mistral", "llama2", "codellama", "phi"]
            
            for model in models_to_test:
                print(f"\nTesting model: {model}")
                provider = OllamaProvider(model_name=model)
                
                start_time = time.time()
                is_available = provider.is_available()
                duration = time.time() - start_time
                
                if is_available:
                    self.provider = provider
                    self.log_result(f"Ollama Availability ({model})", True, 
                                  f"Service available with model {model}", duration)
                    return True
                else:
                    self.log_result(f"Ollama Availability ({model})", False, 
                                  f"Model {model} not available", duration)
            
            # If no models found, test basic connection
            basic_provider = OllamaProvider()
            if basic_provider.is_available():
                self.provider = basic_provider
                return True
            
            return False
            
        except Exception as e:
            self.log_result("Ollama Availability", False, f"Error: {e}")
            return False
    
    def test_sync_generation(self) -> bool:
        """Test synchronous text generation"""
        print("\n🔄 Testing Synchronous Generation")
        print("-" * 50)
        
        if not self.provider:
            self.log_result("Sync Generation", False, "No provider available")
            return False
        
        try:
            # Test basic text generation
            request = LLMRequest(
                prompt="What is 2 + 2? Please answer briefly.",
                context={"test": "basic_math"},
                task_type="basic_query",
                temperature=0.1,
                max_tokens=50
            )
            
            start_time = time.time()
            response = self.provider.generate_sync(request)
            duration = time.time() - start_time
            
            if response and response.content:
                content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
                self.log_result("Sync Generation", True, 
                              f"Generated response: {content_preview}", duration)
                
                # Log metadata if available
                if response.metadata:
                    print(f"   Metadata: {response.metadata}")
                
                return True
            else:
                self.log_result("Sync Generation", False, "Empty response received")
                return False
                
        except Exception as e:
            self.log_result("Sync Generation", False, f"Error: {e}")
            return False
    
    async def test_async_generation(self) -> bool:
        """Test asynchronous text generation"""
        print("\n⚡ Testing Asynchronous Generation")
        print("-" * 50)
        
        if not self.provider:
            self.log_result("Async Generation", False, "No provider available")
            return False
        
        try:
            request = LLMRequest(
                prompt="Explain the concept of distributed computing in one sentence.",
                context={"test": "technical_explanation"},
                task_type="explanation",
                temperature=0.2,
                max_tokens=100
            )
            
            start_time = time.time()
            response = await self.provider.generate_async(request)
            duration = time.time() - start_time
            
            if response and response.content:
                content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
                self.log_result("Async Generation", True, 
                              f"Generated response: {content_preview}", duration)
                return True
            else:
                self.log_result("Async Generation", False, "Empty response received")
                return False
                
        except Exception as e:
            self.log_result("Async Generation", False, f"Error: {e}")
            return False
    
    def test_structured_json_response(self) -> bool:
        """Test structured JSON response generation"""
        print("\n📋 Testing Structured JSON Responses")
        print("-" * 50)
        
        if not self.provider:
            self.log_result("JSON Response", False, "No provider available")
            return False
        
        try:
            # Test job scoring scenario
            request = LLMRequest(
                prompt="""Analyze the following job scheduling scenario and provide a JSON response:
                
Job: CPU cores=4, Memory=8GB, Priority=3
Resource: Available CPU=8, Available Memory=16GB, Current utilization=30%

Please respond with JSON containing:
{
  "score": <float between 0-1>,
  "factors": {"cpu_match": <float>, "memory_match": <float>, "load_factor": <float>},
  "recommendation": "<accept|consider|reject>"
}""",
                context={
                    "job": {"cpu_cores": 4, "memory_gb": 8, "priority": 3},
                    "resource": {"available_cpu": 8, "available_memory": 16, "current_utilization": 0.3}
                },
                task_type="job_scoring",
                temperature=0.1,
                max_tokens=200
            )
            
            start_time = time.time()
            response = self.provider.generate_sync(request)
            duration = time.time() - start_time
            
            if response and response.content:
                try:
                    # Try to parse as JSON
                    json_response = json.loads(response.content)
                    self.log_result("JSON Response", True, 
                                  f"Valid JSON response: {json.dumps(json_response, indent=2)}", duration)
                    
                    # Check for expected fields
                    expected_fields = ["score", "factors", "recommendation"]
                    missing_fields = [field for field in expected_fields if field not in json_response]
                    
                    if missing_fields:
                        print(f"   ⚠️ Missing expected fields: {missing_fields}")
                    else:
                        print(f"   ✅ All expected fields present")
                    
                    return True
                    
                except json.JSONDecodeError as e:
                    self.log_result("JSON Response", False, 
                                  f"Invalid JSON response: {response.content[:100]}")
                    return False
            else:
                self.log_result("JSON Response", False, "Empty response received")
                return False
                
        except Exception as e:
            self.log_result("JSON Response", False, f"Error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and fallback responses"""
        print("\n🛡️ Testing Error Handling")
        print("-" * 50)
        
        try:
            # Test with invalid model
            invalid_provider = OllamaProvider(model_name="nonexistent_model_12345")
            
            request = LLMRequest(
                prompt="This should fail gracefully",
                context={},
                task_type="test_error"
            )
            
            start_time = time.time()
            response = invalid_provider.generate_sync(request)
            duration = time.time() - start_time
            
            # Should get fallback response
            if response and response.content:
                self.log_result("Error Handling", True, 
                              f"Fallback response received: {response.content[:50]}...", duration)
                
                # Check if it's a fallback response
                if "fallback" in response.metadata.get("provider", "").lower():
                    print("   ✅ Proper fallback mechanism activated")
                
                return True
            else:
                self.log_result("Error Handling", False, "No fallback response received")
                return False
                
        except Exception as e:
            self.log_result("Error Handling", False, f"Unexpected error: {e}")
            return False
    
    def test_llm_manager_integration(self) -> bool:
        """Test integration with LLM Manager"""
        print("\n🎯 Testing LLM Manager Integration")
        print("-" * 50)
        
        if not self.provider:
            self.log_result("Manager Integration", False, "No provider available")
            return False
        
        try:
            # Register Ollama provider with manager
            # First extend the LLMProvider enum if needed
            if not hasattr(LLMProvider, 'OLLAMA'):
                LLMProvider.OLLAMA = "ollama"
            
            self.manager.register_provider(LLMProvider.OLLAMA, self.provider)
            self.manager.set_default_provider(LLMProvider.OLLAMA)
            
            request = LLMRequest(
                prompt="Test integration with LLM manager. Respond with 'Integration successful'",
                context={},
                task_type="integration_test"
            )
            
            start_time = time.time()
            response = self.manager.generate_sync(request)
            duration = time.time() - start_time
            
            if response and response.content:
                self.log_result("Manager Integration", True, 
                              f"Manager response: {response.content[:50]}...", duration)
                return True
            else:
                self.log_result("Manager Integration", False, "No response from manager")
                return False
                
        except Exception as e:
            self.log_result("Manager Integration", False, f"Error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model information retrieval"""
        print("\n📊 Testing Model Information Retrieval")
        print("-" * 50)
        
        if not self.provider:
            self.log_result("Model Info", False, "No provider available")
            return False
        
        try:
            start_time = time.time()
            model_info = self.provider.get_model_info()
            duration = time.time() - start_time
            
            if model_info:
                self.log_result("Model Info", True, 
                              f"Model info retrieved", duration)
                print(f"   Model info: {json.dumps(model_info, indent=2)}")
                return True
            else:
                self.log_result("Model Info", False, "No model info available")
                return False
                
        except Exception as e:
            self.log_result("Model Info", False, f"Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 Ollama LLM Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("Service Availability", self.test_ollama_availability()),
            ("Sync Generation", self.test_sync_generation()),
            ("Async Generation", await self.test_async_generation()),
            ("JSON Responses", self.test_structured_json_response()),
            ("Error Handling", self.test_error_handling()),
            ("Manager Integration", self.test_llm_manager_integration()),
            ("Model Info", self.test_model_info())
        ]
        
        # Convert async results to sync for processing
        results = []
        for name, result in tests:
            if asyncio.iscoroutine(result):
                result = await result
            results.append((name, result))
        
        # Summary
        print("\n📈 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {name}")
        
        print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 All tests passed! Ollama integration is working correctly.")
        elif passed > 0:
            print(f"\n⚠️ {total-passed} test(s) failed. Some functionality may be limited.")
        else:
            print("\n❌ All tests failed. Please check Ollama installation and setup.")
        
        return passed == total


async def main():
    """Main test execution"""
    tester = OllamaIntegrationTest()
    success = await tester.run_all_tests()
    
    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure Ollama is installed: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start Ollama service: ollama serve")
        print("3. Pull a model: ollama pull mistral")
        print("4. Check if service is running: curl http://localhost:11434/api/tags")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
