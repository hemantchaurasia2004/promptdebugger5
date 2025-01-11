import streamlit as st
import anthropic
from openai import OpenAI
import json
from typing import Dict, List, Tuple

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class InfluenceVisualizer:
    @staticmethod
    def create_highlighted_text(text: str, segments: List[Dict]) -> str:
        """Create HTML for highlighted text based on influence segments"""
        highlighted_text = text
        
        # Sort segments by score in descending order to highlight strongest influences first
        sorted_segments = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
        
        for segment in sorted_segments:
            segment_text = segment.get('segment', '')
            score = segment.get('score', 0)
            
            if segment_text and score > 0:
                # Create a blue highlight with intensity based on score
                color = f"rgba(0, 0, 255, {score * 0.3})"
                highlight_html = f'<span style="background-color: {color};">{segment_text}</span>'
                highlighted_text = highlighted_text.replace(segment_text, highlight_html)
        
        return highlighted_text

class SystemPromptInfluenceAnalyzer:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.visualizer = InfluenceVisualizer()
        
        self.model_providers = {
            "Anthropic": {
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku"
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
        Return your analysis in this exact JSON format:

        {{
            "system_prompt_influences": [
                {{
                    "segment": "exact text from system prompt",
                    "score": 0.8,
                    "explanation": "why this segment influenced the response"
                }}
            ],
            "customer_message_influence": {{
                "score": 0.7,
                "explanation": "how customer message influenced response"
            }}
        }}

        System Prompt:
        {system_prompt}
        
        Previous Customer Message:
        {previous_customer_message}
        
        Agent Response:
        {agent_response}
        """
        
        try:
            if selected_provider == "Anthropic":
                response = self.anthropic_client.messages.create(
                    model=selected_model,
                    max_tokens=4000,
                    messages=[{
                        "role": "user", 
                        "content": analysis_prompt
                    }],
                    temperature=0.2
                )
                response_text = response.content[0].text
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in analyzing AI system prompts. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )
                response_text = response.choices[0].message.content

            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            return None

def display_conversation_interface(analyzer: SystemPromptInfluenceAnalyzer, 
                                system_prompt: str,
                                conversation_log: str,
                                provider: str,
                                model: str):
    """Display the two-column conversation interface"""
    
    col1, col2 = st.columns(2)
    
    # Left column: System Prompt
    with col1:
        st.header("System Prompt")
        
        if 'current_analysis' in st.session_state and st.session_state.current_analysis:
            highlighted_text = analyzer.visualizer.create_highlighted_text(
                system_prompt,
                st.session_state.current_analysis.get('system_prompt_influences', [])
            )
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Display customer influence if exists
            customer_influence = st.session_state.current_analysis.get('customer_message_influence', {})
            if customer_influence.get('score', 0) > 0:
                st.markdown("### Relevant Customer Context")
                st.markdown(f"""
                Influence Score: {customer_influence.get('score', 0)}
                {customer_influence.get('explanation', '')}
                """)
        else:
            st.markdown(system_prompt)
    
    # Right column: Conversation
    with col2:
        st.header("Conversation")
        messages = analyzer.parse_conversation(conversation_log)
        
        for idx, message in enumerate(messages):
            with st.container():
                st.markdown(f"### {message.role}")
                st.markdown(f"```\n{message.content}\n```")
                
                if message.role == "Agent":
                    if st.button(f"Analyze Response {idx}", key=f"analyze_{idx}"):
                        previous_message = messages[idx-1].content if idx > 0 and messages[idx-1].role == "Customer" else ""
                        
                        with st.spinner("Analyzing response..."):
                            analysis = analyzer.analyze_single_response(
                                system_prompt,
                                message.content,
                                previous_message,
                                provider,
                                model
                            )
                            if analysis:
                                st.session_state.current_analysis = analysis

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
        options=list(analyzer.model_providers.keys()),
        key="provider_select"
    )
    model = st.sidebar.selectbox(
        "Select Model",
        options=list(analyzer.model_providers[provider].keys()),
        format_func=lambda x: analyzer.model_providers[provider][x],
        key="model_select"
    )
    
    # File upload or direct input
    use_direct_input = st.checkbox("Use direct text input", key="use_direct_input")
    
    if use_direct_input:
        system_prompt = st.text_area("System Prompt", height=200, key="system_prompt_input")
        conversation_log = st.text_area("Conversation Log", height=200, key="conversation_log_input")
    else:
        system_prompt_file = st.file_uploader("Upload System Prompt", type=['txt'], key="system_prompt_file")
        conversation_log_file = st.file_uploader("Upload Conversation Log", type=['txt'], key="conversation_log_file")
        
        if system_prompt_file and conversation_log_file:
            system_prompt = system_prompt_file.getvalue().decode('utf-8')
            conversation_log = conversation_log_file.getvalue().decode('utf-8')
        else:
            system_prompt = ""
            conversation_log = ""
    
    if system_prompt and conversation_log:
        display_conversation_interface(analyzer, system_prompt, conversation_log, provider, model)
    else:
        st.warning("Please provide both system prompt and conversation log to continue")

if __name__ == "__main__":
    main()
