import streamlit as st
import anthropic
from openai import OpenAI
import json
from typing import Dict, List, Tuple

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class SystemPromptInfluenceAnalyzer:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        self.model_providers = {
            "Anthropic": {
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku",
                "claude-3-5-haiku-latest": "Claude 3.5 Haiku",
                "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"
            },
            "OpenAI": {
                "gpt-4-0125-preview": "GPT-4 Turbo",
                "gpt-4": "GPT-4",
                "gpt-3.5-turbo": "GPT-3.5 Turbo"
            }
        }

    def parse_conversation(self, conversation_log: str) -> List[Message]:
        """Parse conversation log into structured messages"""
        messages = []
        current_role = None
        current_content = []
        
        for line in conversation_log.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('Customer:', 'Agent:')):
                if current_role:
                    messages.append(Message(current_role, '\n'.join(current_content)))
                    current_content = []
                current_role = line.split(':')[0]
                current_content.append(line.split(':', 1)[1].strip())
            else:
                current_content.append(line)
                
        if current_role and current_content:
            messages.append(Message(current_role, '\n'.join(current_content)))
            
        return messages

    def analyze_single_response(
        self,
        system_prompt: str,
        agent_response: str,
        previous_customer_message: str,
        selected_provider: str,
        selected_model: str
    ) -> Dict:
        """Analyze influence for a single agent response"""
        analysis_prompt = f"""
        Analyze how the system prompt and previous customer message influenced this specific agent response.
        
        System Prompt:
        {system_prompt}
        
        Previous Customer Message:
        {previous_customer_message}
        
        Agent Response:
        {agent_response}
        
        Provide a JSON response with:
        1. system_prompt_influences: List of objects containing:
           - segment: exact text from system prompt
           - score: influence score (0-1.0)
           - explanation: why this segment influenced the response
        2. customer_message_influence: Object containing:
           - score: influence score (0-1.0)
           - explanation: how customer message influenced response
        
        Example format:
        {
            "system_prompt_influences": [
                {
                    "segment": "exact text",
                    "score": 0.8,
                    "explanation": "explanation text"
                }
            ],
            "customer_message_influence": {
                "score": 0.7,
                "explanation": "explanation text"
            }
        }
        """
        
        try:
            if selected_provider == "Anthropic":
                response = self.anthropic_client.messages.create(
                    model=selected_model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                analysis = json.loads(response.content[0].text)
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in analyzing AI system prompts and their influence on conversations."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )
                analysis = json.loads(response.choices[0].message.content)
                
            return analysis
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def render_conversation_message(message: Message, index: int, analyzer: SystemPromptInfluenceAnalyzer,
                              system_prompt: str, messages: List[Message],
                              selected_provider: str, selected_model: str):
    """Render a single conversation message with analysis capability for agent messages"""
    
    if message.role == "Agent":
        with st.container():
            st.markdown("### Agent Response")
            message_container = st.container()
            with message_container:
                st.markdown(f"```\n{message.content}\n```")
                
            if st.button(f"Analyze Response {index}"):
                previous_customer_message = ""
                if index > 0 and messages[index-1].role == "Customer":
                    previous_customer_message = messages[index-1].content
                    
                with st.spinner("Analyzing response..."):
                    analysis = analyzer.analyze_single_response(
                        system_prompt,
                        message.content,
                        previous_customer_message,
                        selected_provider,
                        selected_model
                    )
                    
                if analysis:
                    # Update session state with analysis results
                    st.session_state.current_analysis = analysis
                    st.experimental_rerun()
    else:  # Customer message
        st.markdown("### Customer Message")
        st.markdown(f"```\n{message.content}\n```")

def highlight_system_prompt(system_prompt: str, analysis: Dict) -> None:
    """Render system prompt with highlighted segments based on influence scores"""
    if not analysis:
        st.markdown(system_prompt)
        return
        
    highlighted_text = system_prompt
    for influence in analysis["system_prompt_influences"]:
        segment = influence["segment"]
        score = influence["score"]
        
        # Convert score to RGB color (darker blue for higher influence)
        color = f"rgb({int(255 * (1-score))}, {int(255 * (1-score))}, 255)"
        
        highlighted_text = highlighted_text.replace(
            segment,
            f'<span style="background-color: {color};">{segment}</span>'
        )
    
    st.markdown(highlighted_text, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st.title("Interactive System Prompt Analyzer")
    
    # Initialize analyzer
    analyzer = SystemPromptInfluenceAnalyzer()
    
    # Initialize session state
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    
    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    provider = st.sidebar.selectbox(
        "Select Provider",
        options=list(analyzer.model_providers.keys())
    )
    model = st.sidebar.selectbox(
        "Select Model",
        options=list(analyzer.model_providers[provider].keys()),
        format_func=lambda x: analyzer.model_providers[provider][x]
    )
    
    # File upload or direct input
    use_direct_input = st.checkbox("Use direct text input")
    
    if use_direct_input:
        system_prompt = st.text_area("System Prompt", height=200)
        conversation_log = st.text_area("Conversation Log", height=200)
    else:
        system_prompt_file = st.file_uploader("Upload System Prompt", type=['txt'])
        conversation_log_file = st.file_uploader("Upload Conversation Log", type=['txt'])
        
        if system_prompt_file and conversation_log_file:
            system_prompt = system_prompt_file.getvalue().decode('utf-8')
            conversation_log = conversation_log_file.getvalue().decode('utf-8')
        else:
            st.warning("Please upload both files to continue")
            return
    
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("System Prompt")
        if st.session_state.current_analysis:
            highlight_system_prompt(system_prompt, st.session_state.current_analysis)
            
            if st.session_state.current_analysis["customer_message_influence"]["score"] > 0:
                st.markdown("### Relevant Customer Context")
                st.markdown(f"""
                Influence Score: {st.session_state.current_analysis["customer_message_influence"]["score"]}
                
                {st.session_state.current_analysis["customer_message_influence"]["explanation"]}
                """)
        else:
            st.markdown(system_prompt)
    
    with col2:
        st.header("Conversation")
        messages = analyzer.parse_conversation(conversation_log)
        
        for idx, message in enumerate(messages):
            render_conversation_message(
                message, idx, analyzer, system_prompt, messages,
                provider, model
            )

if __name__ == "__main__":
    main()