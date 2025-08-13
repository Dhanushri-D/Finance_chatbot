import streamlit as st
import os
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    user_type: str
    name: str
    age: int
    income: float
    expenses: Dict[str, float]
    financial_goals: List[str]

class SecurityManager:
    """Handles security features including rate limiting and input validation"""
    
    def __init__(self):
        self.rate_limits = {}
        self.max_requests_per_hour = 50
    
    def hash_user_id(self, user_input: str) -> str:
        """Create anonymous user hash for rate limiting"""
        return hashlib.sha256(user_input.encode()).hexdigest()[:10]
    
    def check_rate_limit(self, user_hash: str) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.now()
        if user_hash not in self.rate_limits:
            self.rate_limits[user_hash] = []
        
        # Remove requests older than 1 hour
        self.rate_limits[user_hash] = [
            req_time for req_time in self.rate_limits[user_hash] 
            if now - req_time < timedelta(hours=1)
        ]
        
        if len(self.rate_limits[user_hash]) >= self.max_requests_per_hour:
            return False
        
        self.rate_limits[user_hash].append(now)
        return True
    
    def sanitize_input(self, text: str) -> str:
        """Clean and validate user input"""
        # Remove potentially harmful characters
        text = re.sub(r'[<>"\';]', '', text)
        # Limit length
        text = text[:1000]
        return text.strip()
    
    def validate_financial_data(self, amount: float) -> bool:
        """Validate financial amounts"""
        return 0 <= amount <= 10000000  # Reasonable limits

class FinanceAI:
    """Core AI processing for financial advice"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load IBM Granite model from HuggingFace"""
        try:
            model_name = "ibm-granite/granite-3b-code-instruct"  # Using available Granite model
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Create pipeline for easier inference
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=500,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a lighter model
            try:
                self.generator = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    device=-1
                )
                logger.info("Loaded fallback model")
            except:
                self.generator = None
                logger.error("Failed to load any model")
    
    def classify_intent(self, user_input: str) -> str:
        """Classify user intent from their message"""
        user_input_lower = user_input.lower()
        
        intents = {
            'budget': ['budget', 'expense', 'spending', 'money', 'cost'],
            'savings': ['save', 'saving', 'savings', 'emergency fund'],
            'investment': ['invest', 'investment', 'stock', 'portfolio', 'retirement'],
            'taxes': ['tax', 'taxes', 'deduction', 'irs', 'filing'],
            'debt': ['debt', 'loan', 'credit', 'payment', 'owe'],
            'general': ['advice', 'help', 'guidance', 'tips']
        }
        
        for intent, keywords in intents.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def create_prompt(self, user_profile: UserProfile, user_question: str, intent: str) -> str:
        """Create contextual prompt based on user profile and intent"""
        
        tone_guide = {
            'student': "Use simple language, be encouraging, focus on practical tips for limited budgets. Avoid complex financial jargon.",
            'professional': "Use professional language, provide detailed analysis, include advanced strategies and technical terms where appropriate.",
            'general': "Use clear, accessible language suitable for a general audience."
        }
        
        tone = tone_guide.get(user_profile.user_type.lower(), tone_guide['general'])
        
        context = f"""
You are a professional financial advisor AI assistant. 

User Profile:
- Type: {user_profile.user_type}
- Age: {user_profile.age}
- Monthly Income: ${user_profile.income:,.2f}
- Financial Goals: {', '.join(user_profile.financial_goals)}

Communication Style: {tone}

Intent: {intent}

Please provide helpful, accurate financial advice for the following question. Be specific, actionable, and tailor your response to the user's profile.

Question: {user_question}

Response:"""
        
        return context
    
    def generate_response(self, prompt: str) -> str:
        """Generate AI response using the loaded model"""
        if not self.generator:
            return "I apologize, but I'm currently unable to process your request. Please try again later."
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=400,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the new response part
            response_text = generated_text.replace(prompt, "").strip()
            
            return response_text if response_text else "I'd be happy to help with your financial question. Could you please provide more details?"
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an issue processing your request. Please try rephrasing your question."

class BudgetAnalyzer:
    """Analyze and visualize budget data"""
    
    def analyze_budget(self, income: float, expenses: Dict[str, float]) -> Dict:
        """Analyze budget and provide insights"""
        total_expenses = sum(expenses.values())
        savings = income - total_expenses
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        # Calculate expense percentages
        expense_percentages = {
            category: (amount / income * 100) if income > 0 else 0
            for category, amount in expenses.items()
        }
        
        # Generate insights
        insights = []
        
        if savings_rate < 10:
            insights.append("⚠️ Your savings rate is below 10%. Consider reviewing your expenses.")
        elif savings_rate >= 20:
            insights.append("✅ Great job! You're saving over 20% of your income.")
        
        # Check housing costs (should be <30% of income)
        housing_expenses = expenses.get('housing', 0) + expenses.get('rent', 0)
        housing_percentage = (housing_expenses / income * 100) if income > 0 else 0
        
        if housing_percentage > 30:
            insights.append("🏠 Housing costs exceed 30% of income. Consider ways to reduce housing expenses.")
        
        return {
            'total_income': income,
            'total_expenses': total_expenses,
            'net_savings': savings,
            'savings_rate': savings_rate,
            'expense_percentages': expense_percentages,
            'insights': insights
        }
    
    def create_budget_chart(self, expenses: Dict[str, float]) -> go.Figure:
        """Create pie chart for expense breakdown"""
        if not expenses:
            return None
        
        fig = px.pie(
            values=list(expenses.values()),
            names=list(expenses.keys()),
            title="Monthly Expense Breakdown"
        )
        fig.update_layout(showlegend=True)
        return fig

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'security_manager' not in st.session_state:
        st.session_state.security_manager = SecurityManager()
    if 'finance_ai' not in st.session_state:
        st.session_state.finance_ai = FinanceAI()
    if 'budget_analyzer' not in st.session_state:
        st.session_state.budget_analyzer = BudgetAnalyzer()

def main():
    st.set_page_config(
        page_title="Personal Finance Chatbot",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("💰 Personal Finance Chatbot")
    st.markdown("*Your AI-powered financial advisor*")
    
    # Sidebar for user profile
    with st.sidebar:
        st.header("👤 User Profile")
        
        # Profile setup
        user_type = st.selectbox(
            "I am a:",
            ["Student", "Professional", "Retiree", "Entrepreneur"]
        )
        
        name = st.text_input("Name (optional)", placeholder="Your name")
        age = st.slider("Age", 18, 80, 25)
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0.0,
            max_value=100000.0,
            value=3000.0,
            step=100.0
        )
        
        # Financial goals
        st.subheader("🎯 Financial Goals")
        goals = st.multiselect(
            "Select your goals:",
            ["Emergency Fund", "Retirement", "House Down Payment", 
             "Debt Payoff", "Investment Growth", "Education Fund"]
        )
        
        # Budget tracking
        st.subheader("📊 Monthly Expenses")
        expenses = {}
        expense_categories = [
            "Housing/Rent", "Food", "Transportation", "Healthcare",
            "Entertainment", "Shopping", "Utilities", "Insurance"
        ]
        
        for category in expense_categories:
            amount = st.number_input(
                f"{category} ($)",
                min_value=0.0,
                max_value=10000.0,
                value=0.0,
                key=f"expense_{category}"
            )
            if amount > 0:
                expenses[category] = amount
        
        # Create user profile
        st.session_state.user_profile = UserProfile(
            user_type=user_type,
            name=name or "User",
            age=age,
            income=monthly_income,
            expenses=expenses,
            financial_goals=goals
        )
        
        # Budget analysis
        if expenses and monthly_income > 0:
            st.subheader("📈 Budget Analysis")
            analysis = st.session_state.budget_analyzer.analyze_budget(
                monthly_income, expenses
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Savings Rate", f"{analysis['savings_rate']:.1f}%")
            with col2:
                st.metric("Monthly Savings", f"${analysis['net_savings']:.2f}")
            
            # Display insights
            for insight in analysis['insights']:
                st.info(insight)
            
            # Chart
            chart = st.session_state.budget_analyzer.create_budget_chart(expenses)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
    
    # Main chat interface
    st.header("💬 Chat with your Financial Advisor")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about budgeting, saving, investing, or any financial question..."):
        # Security checks
        user_hash = st.session_state.security_manager.hash_user_id(prompt)
        
        if not st.session_state.security_manager.check_rate_limit(user_hash):
            st.error("Rate limit exceeded. Please wait before sending more messages.")
            return
        
        # Sanitize input
        clean_prompt = st.session_state.security_manager.sanitize_input(prompt)
        
        if not clean_prompt:
            st.error("Invalid input. Please try again.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Classify intent
                intent = st.session_state.finance_ai.classify_intent(clean_prompt)
                
                # Create prompt
                ai_prompt = st.session_state.finance_ai.create_prompt(
                    st.session_state.user_profile, clean_prompt, intent
                )
                
                # Generate response
                response = st.session_state.finance_ai.generate_response(ai_prompt)
                
                st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Additional features
    st.header("🔧 Additional Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Budget Report"):
            if st.session_state.user_profile.expenses:
                analysis = st.session_state.budget_analyzer.analyze_budget(
                    st.session_state.user_profile.income,
                    st.session_state.user_profile.expenses
                )
                
                st.subheader("Budget Analysis Report")
                st.json(analysis)
            else:
                st.info("Please enter your expenses in the sidebar first.")
    
    with col2:
        if st.button("💡 Get Saving Tips"):
            saving_prompt = f"Give me 5 specific saving tips for a {st.session_state.user_profile.user_type.lower()} with ${st.session_state.user_profile.income:,.2f} monthly income."
            
            intent = "savings"
            ai_prompt = st.session_state.finance_ai.create_prompt(
                st.session_state.user_profile, saving_prompt, intent
            )
            
            response = st.session_state.finance_ai.generate_response(ai_prompt)
            
            st.info(response)
    
    with col3:
        if st.button("🎯 Investment Advice"):
            investment_prompt = f"What investment strategies would you recommend for someone like me?"
            
            intent = "investment"
            ai_prompt = st.session_state.finance_ai.create_prompt(
                st.session_state.user_profile, investment_prompt, intent
            )
            
            response = st.session_state.finance_ai.generate_response(ai_prompt)
            
            st.info(response)
    
    # Footer
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer**: This chatbot provides general financial information only. Always consult with qualified financial professionals for personalized advice.")

if __name__ == "__main__":
    main()