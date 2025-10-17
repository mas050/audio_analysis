import streamlit as st
import google.generativeai as genai
from datetime import datetime
import os
import tempfile
import time
import io
import base64

def initialize_genai(api_key):
    """Initialize the Gemini AI model."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def get_analysis_prompt(analysis_type):
    """Return a specific prompt based on the selected analysis type."""
    base_type = analysis_type.split(" - ")[0]
    
    prompts = {
        "Transcript & Summary": """Please provide both a clean, accurate transcript of this audio file AND a comprehensive summary of the content.

        PART 1 - TRANSCRIPT:
        Please provide a clean, accurate transcript of this audio file. Do not try to associate names that may be mentioned in the audio to piece of text in the transcript.

        PART 2 - SUMMARY:
        Please provide a comprehensive meeting summary with the following elements:

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
        
        Please organize this information in a clear, concise format while maintaining the natural flow of the discussion.
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

def get_full_context_prompt(analysis_type):
    """Get the full context prompt for combined transcripts."""
    base_type = analysis_type.split(" - ")[0]
    
    prompts = {
        "Transcript & Summary": """I will provide you with the combined transcripts from different segments of a long audio file. 
        Please analyze all these transcripts together as a single continuous conversation.
        
        Focus ONLY on creating a COMPREHENSIVE SUMMARY with the following elements:
        
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
        
        DO NOT include the transcript in your response - I will handle adding it separately.
        Just focus on creating the best possible summary of the content.
        
        Here are the combined transcripts:
        
        """,
        
        "Transcription": """I will provide you with the combined transcripts from different segments of a long audio file. 
        Please compile these into a single, clean, accurate transcript of the complete audio, maintaining the flow as if it were one continuous transcription:
        
        """,
        
        "Summary": """I will provide you with the combined transcripts from different segments of a long audio file.
        Please analyze all these transcripts together and provide a comprehensive summary of the entire content, including:
        - Main topics discussed across the entire recording
        - Key points and takeaways
        - Overall context and purpose
        Keep the summary clear and concise while capturing all important information from the entire recording.
        
        Here are the combined transcripts:
        
        """,
        
        "Meeting Summary": """I will provide you with the combined transcripts from different segments of a long audio file.
        Please analyze all these transcripts together as a single continuous meeting and provide a comprehensive meeting summary with the following elements:

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
        
        Here are the combined transcripts:
        
        """,
                
        "Key Quotes": """I will provide you with the combined transcripts from different segments of a long audio file.
        Please analyze all these transcripts together and extract the most significant and impactful quotes from the entire recording.
        For each quote, provide:
        - The exact quote
        - Who said it (if identifiable)
        - Context around the quote
        
        Here are the combined transcripts:
        
        """,
        
        "Content Analysis": """I will provide you with the combined transcripts from different segments of a long audio file.
        Please analyze all these transcripts together and perform a detailed content analysis of the entire recording, including:
        - Tone and mood analysis
        - Key themes and patterns
        - Notable linguistic features
        - Emotional content
        - Professional vs casual language use
        
        Here are the combined transcripts:
        
        """,
        
        "Action Items": """I will provide you with the combined transcripts from different segments of a long audio file.
        Please analyze all these transcripts together and extract all action items, next steps, and commitments mentioned throughout the entire recording.
        Include:
        - Who is responsible (if mentioned)
        - Deadlines or timeframes (if specified)
        - Priority level (if indicated)
        
        Here are the combined transcripts:
        
        """
    }
    return prompts.get(base_type, "")

def process_audio_segments(audio_file, analysis_type, model, num_segments=2):
    """Process audio by sending it in segments to Gemini model."""
    try:
        # Create a progress bar and status text
        progress = st.progress(0)
        status_text = st.empty()
        
        # For longer audio, send in parts and collect transcripts
        status_text.text("Processing audio in segments...")
        
        # Save audio to a temporary file to get direct file access
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + audio_file.name.split('.')[-1])
        temp.write(audio_file.getvalue())
        temp.close()
        
        # Process in segments directly
        transcripts = []
        segment_texts = []
        
        for i in range(num_segments):
            status_text.text(f"Processing segment {i+1}/{num_segments}...")
            
            # We're going to send the entire file, but with instructions to process a specific segment
            segment_prompt = f"""Please transcribe only the {ordinal(i+1)} segment of this audio (approximately from {i/num_segments:.0%} to {(i+1)/num_segments:.0%} of the total duration).
            Focus only on this portion of the audio and ignore the rest."""
            
            # Read file as bytes and create Part object
            with open(temp.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Create audio part with inline data
            file_ext = audio_file.name.split('.')[-1].lower()
            mime_types = {
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'm4a': 'audio/mp4'
            }
            mime_type = mime_types.get(file_ext, 'audio/mpeg')
            
            audio_part = {
                'mime_type': mime_type,
                'data': audio_bytes
            }
            
            response = model.generate_content([audio_part, segment_prompt])
            
            segment_text = response.text
            segment_texts.append(segment_text)
            transcripts.append(f"--- SEGMENT {i+1}/{num_segments} TRANSCRIPT ---\n{segment_text}")
            progress.progress((i + 1) / (num_segments + 1))
        
        # Clean up the temporary file
        try:
            os.unlink(temp.name)
        except:
            st.warning(f"Could not remove temporary file: {temp.name}")
        
        # Build a clean full transcript
        full_transcript = "\n\n".join(segment_texts)
        
        # Process all transcripts for the final analysis - but don't include transcript in the result
        status_text.text("Generating final analysis from all segments...")
        summary_result = process_transcripts(transcripts, analysis_type, model)
        progress.progress(1.0)
        status_text.text("Analysis complete!")
        
        if analysis_type.startswith("Transcript & Summary"):
            # For transcript & summary type, manually combine transcript and summary
            final_result = f"""# COMPLETE TRANSCRIPT

{full_transcript}

# COMPREHENSIVE SUMMARY

{summary_result}"""
            return final_result
        else:
            # For other types, just return the LLM output
            return summary_result
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return f"Error processing audio: {str(e)}"

def ordinal(n):
    """Return the ordinal representation of a number."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def process_transcripts(transcripts, analysis_type, model):
    """Process the combined transcripts with the final analysis."""
    try:
        # Combine all transcripts with the full context prompt
        full_prompt = get_full_context_prompt(analysis_type) + "\n".join(transcripts)
        
        # Send to Gemini for final analysis with appropriate configuration
        response = model.generate_content(full_prompt, 
                                          generation_config=genai.types.GenerationConfig(
                                              temperature=0.2,  # Lower temperature for more precise output
                                              max_output_tokens=16000  # Allow enough space for detailed summary
                                          ))
        return response.text
    except Exception as e:
        return f"Error processing combined transcripts: {str(e)}"

def process_audio(audio_file, analysis_type, model, use_segmentation=False, num_segments=2):
    """Process the audio file with or without segmentation based on user selection."""
    if use_segmentation:
        return process_audio_segments(audio_file, analysis_type, model, num_segments)
    else:
        try:
            # Create a temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + audio_file.name.split('.')[-1])
            tmp_file_path = tmp_file.name
            tmp_file.write(audio_file.getvalue())
            tmp_file.close()  # Close the file handle immediately

            try:
                # Read file as bytes and create Part object
                with open(tmp_file_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Create audio part with inline data
                file_ext = audio_file.name.split('.')[-1].lower()
                mime_types = {
                    'mp3': 'audio/mpeg',
                    'wav': 'audio/wav',
                    'm4a': 'audio/mp4'
                }
                mime_type = mime_types.get(file_ext, 'audio/mpeg')
                
                audio_part = {
                    'mime_type': mime_type,
                    'data': audio_bytes
                }
                
                prompt = get_analysis_prompt(analysis_type)
                response = model.generate_content([audio_part, prompt])
                result = response.text
                
                # For transcript & summary type without segmentation, we need to handle it specially
                if analysis_type.startswith("Transcript & Summary"):
                    # Extract parts
                    result_parts = result.split("PART 2 - SUMMARY")
                    if len(result_parts) == 2:
                        transcript_part = result_parts[0].replace("PART 1 - TRANSCRIPT:", "").strip()
                        summary_part = "PART 2 - SUMMARY" + result_parts[1].strip()
                        result = f"""# COMPLETE TRANSCRIPT

{transcript_part}

# COMPREHENSIVE SUMMARY

{summary_part}"""
                
            finally:
                # Add a small delay before trying to remove the file
                time.sleep(0.5)
                if os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception as cleanup_error:
                        st.warning(f"Warning: Could not remove temporary file {tmp_file_path}: {cleanup_error}")
            
            return result
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"

def main():
    st.title("Advanced Audio Analysis Tool")
    
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
            
            # Option for processing longer files
            use_segmentation = st.checkbox("Process as long audio (split into segments)", 
                              help="Use this option for files longer than 1 hour to improve processing")
            
            # Number of segments
            num_segments = 2  # Default value
            if use_segmentation:
                num_segments = st.slider("Number of segments", min_value=2, max_value=8, value=2, 
                                        help="More segments allows for longer audio, but may reduce context between segments")
                st.info(f"📌 Long audio mode will process your file in {num_segments} equal segments, then combine results. This allows processing of much longer files than the model can handle directly.")
            
            # Process audio button
            if audio_file and st.button("Analyze Audio"):
                with st.spinner("Processing audio..."):
                    st.session_state.analysis_result = process_audio(audio_file, selected_type, model, use_segmentation, num_segments)
            
            # Display results if available
            if st.session_state.analysis_result:
                st.subheader("Analysis Results")
                st.text_area("Output", st.session_state.analysis_result, height=300)
                
                # Download button in a separate column
                col1, col2 = st.columns([1, 4])
                with col1:
                    # Add download button that uses the uploaded file's name
                    from datetime import datetime
                    import os
                    current_time = datetime.now().strftime("%Y%m%d_%H%M")
                    
                    # Get the original filename without extension and replace spaces with underscores
                    original_filename = os.path.splitext(audio_file.name)[0]
                    original_filename = original_filename.replace(" ", "_")
                    
                    # Create new filename with original file name and timestamp
                    filename = f"{original_filename}_{current_time}.txt"
                    
                    st.download_button(
                        label="💾 Download",
                        data=st.session_state.analysis_result,
                        file_name=filename,
                        mime="text/plain"
                    )
                
        except Exception as e:
            st.error(f"Error initializing Gemini AI: {str(e)}")
    else:
        st.warning("Please enter your Gemini API key to proceed")

if __name__ == "__main__":
    main()
