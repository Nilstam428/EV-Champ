import os
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        print("🚀 Initializing LLMService...")
        self.client = None
        self.rag_model = None
        self.chat_model = None
        self.initialization_error = None
        
        self._initialize_client()
        print("✅ LLMService initialization complete")
    
    def _initialize_client(self):
        """Initialize the Groq client with comprehensive error handling"""
        print("🔧 Starting client initialization...")
        try:
            if not settings.GROQ_API_KEY:
                print("❌ GROQ_API_KEY is missing!")
                raise ValueError("GROQ_API_KEY is required")
            
            print("🔑 GROQ_API_KEY found, length:", len(settings.GROQ_API_KEY))
            
            # Check if there are any proxy-related settings in the environment
            print("🔍 Checking for any proxy-related configurations...")
            proxy_related_vars = [
                'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'
            ]
            
            current_proxies = {}
            for var in proxy_related_vars:
                if var in os.environ:
                    current_proxies[var] = os.environ[var]
                    print(f"   Found {var}: {os.environ[var][:50]}...")
            
            if not current_proxies:
                print("   No proxy environment variables found")
            
            # Clean environment approach - temporarily remove proxy settings
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
            original_values = {}
            
            try:
                # Save and temporarily clear proxy environment variables
                print("🌐 Temporarily clearing proxy environment variables...")
                for var in proxy_vars:
                    if var in os.environ:
                        original_values[var] = os.environ[var]
                        del os.environ[var]
                        print(f"   Temporarily removed {var}")
                
                if not original_values:
                    print("   No proxy variables to remove")
                
                # Also clear any other potentially problematic environment variables
                other_vars_to_clear = ['REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE']
                other_original_values = {}
                for var in other_vars_to_clear:
                    if var in os.environ:
                        other_original_values[var] = os.environ[var]
                        del os.environ[var]
                        print(f"   Temporarily removed {var}")
                
                # Try different initialization approaches
                print("🔄 Attempting client initialization...")
                self.client = self._try_client_initialization()
                
            finally:
                # Restore original proxy settings
                print("🔄 Restoring environment variables...")
                for var, value in original_values.items():
                    os.environ[var] = value
                    print(f"   Restored {var}")
                for var, value in other_original_values.items():
                    os.environ[var] = value
                    print(f"   Restored {var}")
            
            # Set model names
            self.rag_model = getattr(settings, 'RAG_MODEL_NAME', 'llama3-8b-8192')
            self.chat_model = getattr(settings, 'CHAT_MODEL_NAME', 'llama3-8b-8192')
            
            print(f"🤖 RAG Model: {self.rag_model}")
            print(f"💬 Chat Model: {self.chat_model}")
            
            logger.info("LLM Service initialized successfully")
            print("✅ LLM Service initialized successfully")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Failed to initialize LLM Service: {e}")
            print(f"❌ Failed to initialize LLM Service: {e}")
            # Don't raise here - allow graceful degradation
    
    def _try_client_initialization(self) -> Groq:
        """Try different approaches to initialize the Groq client"""
        print("🔄 Trying multiple client initialization methods...")
        
        # Get clean kwargs without proxy-related parameters
        def get_clean_kwargs(**kwargs):
            """Remove any proxy-related or unsupported arguments"""
            clean_kwargs = {}
            supported_params = ['api_key', 'timeout', 'max_retries', 'base_url']
            
            for key, value in kwargs.items():
                if key in supported_params and value is not None:
                    clean_kwargs[key] = value
            
            print(f"   Clean kwargs: {list(clean_kwargs.keys())}")
            return clean_kwargs
        
        initialization_methods = [
            # Method 1: Only API key
            lambda: Groq(**get_clean_kwargs(api_key=settings.GROQ_API_KEY)),
            
            # Method 2: API key with timeout
            lambda: Groq(**get_clean_kwargs(
                api_key=settings.GROQ_API_KEY,
                timeout=30.0
            )),
            
            # Method 3: API key with max retries
            lambda: Groq(**get_clean_kwargs(
                api_key=settings.GROQ_API_KEY,
                max_retries=3
            )),
            
            # Method 4: Minimal - just pass api_key directly
            lambda: Groq(api_key=settings.GROQ_API_KEY),
        ]
        
        last_error = None
        for i, method in enumerate(initialization_methods):
            try:
                print(f"   Trying method {i + 1}...")
                client = method()
                logger.info(f"Client initialized using method {i + 1}")
                print(f"✅ Client initialized successfully using method {i + 1}")
                return client
            except Exception as e:
                last_error = e
                logger.warning(f"Initialization method {i + 1} failed: {e}")
                print(f"❌ Method {i + 1} failed: {e}")
                continue
        
        # If all methods fail, raise the last error
        print("❌ All initialization methods failed!")
        raise last_error or Exception("All initialization methods failed")
    
    def _format_chat_history(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Format chat history for Groq API"""
        print("📝 Formatting chat history...")
        print(f"   Input messages count: {len(messages)}")
        
        formatted_messages = []
        
        # Add system message
        system_message = {
            "role": "system",
             "content": "You are EV Champ, an AI assistant for assisting user is their queiries related to electric vehicle. You help users with electric vehicle operations, maintenance, charging, route optimization, and related queries. Be helpful, professional, and concise."
        }
        formatted_messages.append(system_message)
        print("   Added system message")
        
        # Add chat history (reverse to get chronological order)
        recent_messages = messages[-10:]  # Last 10 messages for context
        print(f"   Using last {len(recent_messages)} messages for context")
        
        for i, msg in enumerate(reversed(recent_messages)):
            role = "user" if msg.is_user else "assistant"
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
            print(f"   Message {i+1}: {role} - {len(msg.content)} characters")
        
        print(f"✅ Formatted {len(formatted_messages)} total messages")
        return formatted_messages
    
    def _get_rag_context(self, query: str) -> str:
        """
        Placeholder for RAG implementation
        In production, this would:
        1. Embed the query
        2. Search vector database
        3. Retrieve relevant documents
        4. Return context
        """
        print("🔍 Getting RAG context...")
        print(f"   Query: {query[:50]}...")
        
        # Mock RAG context for EV Fleet Management
        rag_context = """
        EV Fleet Management Context:
        - Electric vehicles require regular battery health monitoring
        - Optimal charging occurs between 20-80% battery level
        - Route planning should consider charging station locations
        - Fleet maintenance includes battery, tires, and software updates
        - Energy consumption varies by weather, load, and driving patterns
        """
        
        print(f"✅ RAG context retrieved ({len(rag_context)} characters)")
        return rag_context
    
    def _get_fallback_response(self, query: str, error_msg: str) -> str:
        """Generate a fallback response when LLM service is unavailable"""
        print("⚠️  Generating fallback response...")
        print(f"   Query: {query[:50]}...")
        print(f"   Error: {error_msg}")
        
        fallback_responses = {
            'greeting': "Hello! I'm EV Champ, your your personal EV Chatbot. How can I help you today?",
            'battery': "For battery-related queries, ensure you maintain charge levels between 20-80% for optimal battery health.",
            'charging': "For charging information, consider location, power levels, and charging schedules for your fleet.",
            'maintenance': "Regular maintenance includes battery health checks, tire rotation, and software updates.",
            'route': "Route optimization should consider charging station locations and vehicle range.",
            'default': f"I apologize, but I'm currently experiencing technical difficulties and cannot process your request fully. Please try again later or contact support. Your query: '{query[:50]}...'"
        }
        
        query_lower = query.lower()
        selected_response = 'default'
        
        for key, response in fallback_responses.items():
            if key in query_lower:
                selected_response = key
                break
        
        print(f"   Selected fallback type: {selected_response}")
        print(f"✅ Fallback response generated ({len(fallback_responses[selected_response])} characters)")
        return fallback_responses[selected_response]
    
    async def generate_response(self, query: str, chat_history: List[Any], rag_enabled: bool = False) -> str:
        """Generate response using appropriate model based on RAG setting"""
        print("\n" + "="*50)
        print("🤖 Starting response generation...")
        print(f"   Query: {query[:100]}...")
        print(f"   RAG enabled: {rag_enabled}")
        print(f"   Chat history length: {len(chat_history)}")
        
        try:
            # Check if service was initialized properly
            if self.initialization_error:
                print("❌ Service has initialization error")
                return self._get_fallback_response(query, self.initialization_error)
            
            if not self.client:
                print("❌ Client not initialized")
                return self._get_fallback_response(query, "Client not initialized")
            
            print("✅ Service and client are healthy")
            
            # Format messages
            print("📝 Formatting messages...")
            messages = self._format_chat_history(chat_history)
            
            if rag_enabled:
                print("🔍 RAG mode enabled - enhancing query with context")
                # Use RAG model with enhanced context
                rag_context = self._get_rag_context(query)
                enhanced_query = f"Context: {rag_context}\n\nUser Query: {query}"
                messages.append({"role": "user", "content": enhanced_query})
                model = self.rag_model
                print(f"   Enhanced query length: {len(enhanced_query)} characters")
            else:
                print("💬 Regular chat mode")
                # Use regular chat model
                messages.append({"role": "user", "content": query})
                model = self.chat_model
            
            if not model:
                print("❌ Model not configured")
                return self._get_fallback_response(query, "Model not configured")
            
            print(f"🤖 Using model: {model}")
            
            # Call Groq API with retry logic
            max_retries = 2
            print(f"🔄 Starting API calls (max retries: {max_retries})")
            
            for attempt in range(max_retries):
                try:
                    print(f"   Attempt {attempt + 1}/{max_retries}")
                    print("   Making API call to Groq...")
                    
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    
                    print("✅ API call successful")
                    
                    # Validate response
                    if not response or not hasattr(response, 'choices') or not response.choices:
                        print("⚠️  Invalid response structure")
                        if attempt < max_retries - 1:
                            print("   Retrying...")
                            continue
                        return self._get_fallback_response(query, "Empty response from API")
                    
                    print(f"   Response has {len(response.choices)} choices")
                    
                    content = response.choices[0].message.content
                    if not content:
                        print("⚠️  No content in response")
                        return self._get_fallback_response(query, "No content in response")
                    
                    print(f"✅ Response generated successfully ({len(content)} characters)")
                    print("="*50 + "\n")
                    print(content)
                    return content
                    
                except Exception as api_error:
                    error_message = str(api_error).lower()
                    print(f"❌ API call failed: {api_error}")
                    
                    if attempt < max_retries - 1:
                        # Retry on certain errors
                        if any(retry_error in error_message for retry_error in ['timeout', 'connection', 'temporary']):
                            print("   Error is retryable, continuing...")
                            continue
                        else:
                            print("   Error is not retryable")
                    
                    # Handle specific errors
                    if "rate limit" in error_message:
                        print("   Rate limit error detected")
                        return "I'm currently experiencing high demand. Please try again in a moment."
                    elif "invalid api key" in error_message:
                        print("   Invalid API key error detected")
                        return "There seems to be an issue with the API configuration. Please contact support."
                    elif "model" in error_message and "not found" in error_message:
                        print(f"   Model not found error detected for {model}")
                        return f"The requested model ({model}) is not available. Please contact support."
                    else:
                        print("   Generic API error")
                        return self._get_fallback_response(query, str(api_error))
            
            # If all retries failed
            print("❌ All retries exhausted")
            return self._get_fallback_response(query, "Maximum retries exceeded")
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            print(f"💥 Unexpected error in generate_response: {e}")
            return self._get_fallback_response(query, str(e))
    
    def is_healthy(self) -> bool:
        """Check if the LLM service is healthy"""
        healthy = self.client is not None and self.initialization_error is None
        print(f"🏥 Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
        return healthy
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the LLM service"""
        print("📊 Getting service status...")
        
        status = {
            "healthy": self.is_healthy(),
            "client_initialized": self.client is not None,
            "models_configured": bool(self.rag_model and self.chat_model),
            "rag_model": self.rag_model,
            "chat_model": self.chat_model,
            "initialization_error": self.initialization_error
        }
        
        print("   Status details:")
        for key, value in status.items():
            print(f"   - {key}: {value}")
        
        return status