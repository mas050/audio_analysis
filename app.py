import streamlit as st
import google.generativeai as genai
import pyperclip
from tempfile import NamedTemporaryFile
import os

def initialize_genai(api_key):
    """Initialize the Gemini AI model."""
    genai.configure(api_key=api_key)
    #return genai.GenerativeModel("gemini-1.5-flash")
    #return genai.GenerativeModel("gemini-2.0-flash")
    return genai.GenerativeModel("gemini-2.0-flash-lite")
    
def get_analysis_prompt(analysis_type):
    """Return a specific prompt based on the selected analysis type."""
    base_type = analysis_type.split(" - ")[0]
    
    prompts = {
        "Transcript & Summary": """Please provide both a clean, accurate transcript of this audio file AND a comprehensive summary of the content.

        PART 1 - TRANSCRIPT:
        Please provide a clean, accurate transcript of this audio file. Do not try to associate names that may be mentioned in the audio to piece of text in the transcript.

        PART 2 - SUMMARY:
        Follow the transcript with a comprehensive summary that includes:
        - Main topics discussed
        - Key points and takeaways
        - Overall context and purpose
        - Important conclusions or decisions

        Format your response with clear headings separating the transcript and summary sections.""",
        
        "Transcription": "Please provide a clean, accurate transcript of this audio file.",
        
        "Summary": """Please provide a comprehensive summary of this audio content, including:
        - Main topics discussed
        - Key points and takeaways
        - Overall context and purpose
        Keep the summary clear and concise while capturing all important information.""",
        
        "Meeting Summary": """Please provide a comprehensive meeting summary with the following elements:

        1. Meeting Overview:
        - Identify all participants and their roles
        - Extract the main purpose/objective of the meeting
        - Note the overall tone and engagement level
        
        2. Key Discussion Points:
        - List and elaborate on the major topics discussed
        - Highlight any decisions made or conclusions reached
        - Capture important questions raised and their answers
        
        3. Action Items & Next Steps:
        - Extract all tasks and assignments
        - Include who is responsible for each item
        - Note any mentioned deadlines or timeframes
        - Flag any items marked as high priority
        
        4. Follow-up Requirements:
        - List any scheduled follow-up meetings
        - Note any documents or resources that were requested
        - Identify any pending decisions or unresolved issues
        
        5. Notable Quotes & Key Insights:
        - Extract significant statements or important insights
        - Include context for each notable point
        - Highlight any strategic or innovative ideas proposed
        
        6. Additional Context:
        - Note any important references to past meetings or decisions
        - Capture any mentioned risks or concerns
        - Highlight any budget or resource discussions
        
        Please organize this information in a clear, concise format while maintaining the natural flow of the discussion. Include approximate timestamps for major topic transitions.""",
                
        "Key Quotes": """Extract the most significant and impactful quotes from this audio.
        For each quote, provide:
        - The exact quote
        - Who said it (if identifiable)
        - Context around the quote""",
        
        "Content Analysis": """Perform a detailed content analysis of this audio, including:
        - Tone and mood analysis
        - Key themes and patterns
        - Notable linguistic features
        - Emotional content
        - Professional vs casual language use""",
        
        "Action Items": """Extract all action items, next steps, and commitments mentioned in this audio.
        Include:
        - Who is responsible (if mentioned)
        - Deadlines or timeframes (if specified)
        - Priority level (if indicated)"""
    }
    return prompts.get(base_type)

def process_audio(audio_file, analysis_type, model):
    """Process the audio file with Gemini AI."""
    try:
        with NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name

        gemini_file = genai.upload_file(tmp_file_path)
        prompt = get_analysis_prompt(analysis_type)
        response = model.generate_content([gemini_file, prompt])
        
        os.unlink(tmp_file_path)
        return response.text
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def main():
    st.title("🎧 Audio Analysis Tool")
    
    # Initialize session state for storing results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""
    
    # Analysis options with descriptions included in the options
    analysis_options = [
        "Transcript & Summary - Generate both a complete transcript and a comprehensive summary",
        "Transcription - Convert speech to text with high accuracy",
        "Summary - Generate a concise overview of the main points",
        "Meeting Summary - Create a structured summary with participants, decisions, and action items",
        "Key Quotes - Extract important statements and their context",
        "Content Analysis - Analyze tone, themes, and linguistic patterns",
        "Action Items - Identify all tasks, assignments, and commitments"
    ]
    
    # API Key input
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    if api_key:
        try:
            model = initialize_genai(api_key)
            
            # File uploader
            st.subheader("Upload Audio File")
            audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
            
            # Dropdown for analysis type
            selected_type = st.selectbox(
                "Select Analysis Type",
                analysis_options
            )
            
            # Process audio button
            if audio_file and st.button("Analyze Audio"):
                with st.spinner("Processing audio..."):
                    st.session_state.analysis_result = process_audio(audio_file, selected_type, model)
            
            # Display results if available
            if st.session_state.analysis_result:
                st.subheader("Analysis Results")
                st.text_area("Output", st.session_state.analysis_result, height=300)
                
                # Copy button in a separate column
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("📋 Copy"):
                        try:
                            pyperclip.copy(st.session_state.analysis_result)
                            st.success("Copied!")
                        except Exception as e:
                            st.error(f"Error copying to clipboard: {str(e)}")
                
        except Exception as e:
            st.error(f"Error initializing Gemini AI: {str(e)}")
    else:
        st.warning("Please enter your Gemini API key to proceed")

if __name__ == "__main__":
    main()
