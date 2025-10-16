import streamlit as st
import os
import mimetypes
from google import genai
from google.genai import types
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Function to enhance the prompt using Groq
def make_prompt(simple_prompt, style, color, size, placement):
    """Enhance a simple prompt into a highly detailed and structured one using ChatGroq."""
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.7
    )


    # Define the prompt enhancement template
    system_message = (
        "You are an expert in designing realistic tattoo prompts. "
        "Enhance the given tattoo idea by adding intricate details, shading depth, artistic elements, "
        "and making it fit well for placement on a specific body part."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Tattoo Description: {simple_prompt}\nStyle: {style}\nColor Scheme: {color}\nSize: {size}\nPlacement: {placement}")
    ])

    # Create the enhancement chain and invoke it
    chain = prompt_template | llm
    enhanced_prompt = chain.invoke({
        "simple_prompt": simple_prompt,
        "style": style,
        "color": color,
        "size": size,
        "placement": placement
    })

    return enhanced_prompt.content  # Return the string content directly

# Function to generate tattoo using Gemini API
def generate_tattoo(prompt):
    """
    Generate a tattoo image based on the constructed prompt using the Gemini API.
    
    Args:
        prompt (str): The constructed prompt including description, style, color, size, and placement.
    
    Returns:
        tuple: (result, mime_type) where result is either image bytes or text, 
               and mime_type is the image format or None if text.
    """
    try:
        # Initialize the Gemini API client with the API key from environment variables
        
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        model = "gemini-2.0-flash-exp-image-generation"
        
        # Prepare the content with the constructed prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        # Configure the generation settings
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
        )
        
        image_data = None
        mime_type = None
        text_response = ""
        
        # Stream the response from the API
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data:
                image_data = part.inline_data.data
                mime_type = part.inline_data.mime_type
                break
            else:
                text_response += part.text
        
        if image_data:
            return image_data, mime_type
        else:
            return text_response, None
    except Exception as e:
        return f"Error: {str(e)}", None

# Set up the Streamlit application
st.title("Tattoo Generator")

# Define options for dropdowns
styles = ["Traditional", "Realistic", "Watercolor", "Geometric", "Tribal"]
color_schemes = ["Black and White", "Full Color"]
sizes = ["extra small","Small", "Medium", "Big"]
placements = ["Arm", "Leg", "Back", "Chest", "Neck"]

# Create a form for user input
with st.form(key='tattoo_form'):
    description = st.text_area("Description:", placeholder="e.g., A lion with a crown", key='desc')
    style = st.selectbox("Style:", styles, key='style')
    color_scheme = st.selectbox("Color Scheme:", color_schemes, key='color')
    size = st.selectbox("Tattoo Size:", sizes, key='size')
    placement = st.selectbox("Placement:", placements, key='placement')
    submit_button = st.form_submit_button('Generate Tattoo')

# Handle form submission
if submit_button:
    if st.session_state['desc'].strip() == "":
        st.error("Please enter a description.")
    else:
        with st.spinner("Enhancing prompt and generating tattoo..."):
            # Enhance the user's description
            enhanced_description = make_prompt(
                st.session_state['desc'],
                st.session_state['style'],
                st.session_state['color'],
                st.session_state['size'],
                st.session_state['placement']
            )
            
            # Construct the prompt with explicit body placement            
            prompt = f"Generate an image of a tattoo design featuring {enhanced_description} in {st.session_state['style']} style with {st.session_state['color']} colors,The size should be {st.session_state['size']},realistically placed on the {st.session_state['placement']} of a human body.,give proper discription for size of the tattoo"
            caption = f"Generated Tattoo on {st.session_state['placement']}"

            # Generate the tattoo image
            result, mime_type = generate_tattoo(prompt)
            if mime_type:
                st.image(result, caption=caption, use_container_width=True)
                extension = mimetypes.guess_extension(mime_type)
                file_name = f"tattoo{extension}" if extension else "tattoo"
                st.download_button(
                    label="Download Image",
                    data=result,
                    file_name=file_name,
                    mime=mime_type,
                )
            else:
                st.write("Response:", result)
